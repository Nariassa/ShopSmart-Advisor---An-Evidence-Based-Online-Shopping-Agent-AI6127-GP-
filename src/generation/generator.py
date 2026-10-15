"""Grounded Generation & Purchase Confidence Scoring's public API: synthesis, self-check, citations, PCS and UI payloads."""

from __future__ import annotations

from collections import defaultdict
from html import escape, unescape
import re
import time

from .evidence import canonical_aspect, clean_number, detect_aspects, normalize_evidence
from .grounding import extract_facts, verify_claims, verify_selections
from .llm import LLMUnavailable
from .pcs import PCSConfig, calculate_pcs
from .prompts import bounded_schema, build_selection_prompt, build_english_rewrite_prompt
from .product_match import requirements, filter_products
from .sentiment import LexiconSentiment, evidence_sentiment


def infer_constraints(query):
    match = re.search(r"(?:under|below|less than|budget(?: of)?|不超过|低于|预算)\s*\$\s*(\d+(?:\.\d+)?)", query, re.I)
    return {"max_price": float(match.group(1)), "currency": "USD"} if match else {}


def price_check(items, constraints):
    if "max_price" not in constraints:
        return {"status": "not_requested"}
    maximum = clean_number(constraints["max_price"])
    if maximum is None or maximum < 0:
        raise ValueError("max_price must be a non-negative finite number.")
    requested_currency = str(constraints.get("currency", "USD")).upper()
    prices = []
    for item in items:
        if item.chunk_type != "spec":
            continue
        raw = item.metadata.get("price")
        amount = clean_number(raw)
        currency = item.metadata.get("currency")
        if isinstance(raw, str) and raw.startswith("$"):
            amount = clean_number(raw[1:])
            currency = currency or "USD"
        if amount is not None and amount >= 0 and str(currency).upper() == requested_currency:
            prices.append((amount, item.chunk_id))
    if not prices or len({amount for amount, _ in prices}) > 1:
        return {"status": "unknown", "max_price": maximum, "currency": requested_currency}
    amount, chunk_id = prices[0]
    return {"status": "met" if amount <= maximum else "violated",
            "price": amount, "max_price": maximum, "currency": requested_currency,
            "chunk_id": chunk_id, "scope": "dataset_price_not_live"}


class ShopSmartGenerator:
    def __init__(self, llm=None, sentiment_classifier=None, grounding_verifier=None,
                 pcs_config=None, max_products=3, max_facts=60, max_claims=24,
                 max_context_chars=16000, allow_fallback=True, prompt_version="v1"):
        if min(max_products, max_facts, max_claims, max_context_chars) < 1:
            raise ValueError("Generator limits must be positive.")
        if prompt_version not in {"v1", "v2", "v3"}:
            raise ValueError("Prompt version must be v1, v2 or v3.")
        self.prompt_version = prompt_version
        self.llm = llm
        self.classifier = sentiment_classifier or LexiconSentiment()
        self.verifier = grounding_verifier
        self.pcs_config = pcs_config or PCSConfig()
        self.max_products, self.max_facts, self.max_claims = max_products, max_facts, max_claims
        self.max_context_chars, self.allow_fallback = max_context_chars, allow_fallback

    def generate_recommendation(self, query, evidence_set, *, requested_aspects=None,
                                aspect_weights=None, product_signals=None, constraints=None, now=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be non-empty text.")
        if len(query) > 2000:
            raise ValueError("Query must be at most 2000 characters.")
        started = time.perf_counter()
        evidence = normalize_evidence(evidence_set)
        aspects = list(dict.fromkeys(canonical_aspect(a) for a in
                                    (requested_aspects or detect_aspects(query))))
        constraints = requirements(query, aspects, infer_constraints(query) if constraints is None else constraints)
        evidence, excluded_products = filter_products(evidence, constraints)
        by_product = defaultdict(list)
        sentiments = {}
        for item in evidence:
            by_product[item.parent_asin].append(item)
            sentiments[item.chunk_id] = evidence_sentiment(item, self.classifier)
        products = []
        for asin, items in by_product.items():
            pcs = calculate_pcs(items, sentiments, aspects, aspect_weights=aspect_weights,
                                signals=(product_signals or {}).get(asin), config=self.pcs_config, now=now)
            budget = price_check(items, constraints)
            products.append({"parent_asin": asin, "title": items[0].title, "pcs": pcs,
                             "budget_check": budget,
                             "decision_score": round(pcs["score"] * (0.25 + 0.75 * pcs["fit_score"]), 4)})
        products.sort(key=lambda p: (p["budget_check"]["status"] == "violated",
                                     -p["decision_score"], p["parent_asin"]))
        products = products[:self.max_products]
        selected_asins = {p["parent_asin"] for p in products}
        shown_evidence = [item for item in evidence if item.parent_asin in selected_asins]
        facts = extract_facts(shown_evidence, query, aspects, self.classifier, sentiments,
                              self.max_facts, self.max_context_chars)
        prepared = time.perf_counter()
        rejected, warnings = [], []
        backend = "extractive"
        accepted = []
        llm_verified_claims = 0
        selection_info, rewrite_info = None, None
        llm_selections, selection_rejected = 0, []
        rewrite_verified = 0
        if self.llm is not None and facts:
            try:
                prompt = build_selection_prompt(query, facts, shown_evidence, self.max_claims,
                                               requested_aspects=aspects, constraints=constraints,
                                               version=self.prompt_version)
                if hasattr(self.llm, "generate_structured"):
                    payload = self.llm.generate_structured(prompt, bounded_schema('selections', self.max_claims))
                else:
                    payload = self.llm.generate(prompt)
                selection_info = self.llm.describe() if hasattr(self.llm, "describe") else None
                accepted, selection_rejected = verify_selections(payload, facts, shown_evidence, self.max_claims)
                rejected.extend(selection_rejected)
                llm_selections = len(accepted)
                if self.verifier is not None and accepted:
                    selected_ids = {claim["fact_id"] for claim in accepted}
                    selected_facts = [fact for fact in facts if fact.fact_id in selected_ids]
                    rewrite_prompt = build_english_rewrite_prompt(selected_facts, self.max_claims)
                    try:
                        if hasattr(self.llm, 'generate_structured'):
                            rewritten = self.llm.generate_structured(rewrite_prompt, bounded_schema('claims', self.max_claims))
                        else:
                            rewritten = self.llm.generate(rewrite_prompt)
                        rewrite_info = self.llm.describe() if hasattr(self.llm, "describe") else None
                        verified, failures = verify_claims(rewritten, selected_facts, shown_evidence,
                                                           self.verifier, self.max_claims, english_only=True, strict_rewrite=True)
                        rejected.extend(failures)
                        rewrite_verified = len(verified)
                        by_id = {claim["fact_id"]: claim for claim in verified}
                        # Unsupported rewrites never replace the safe program-hydrated quote.
                        accepted = [by_id.get(claim["fact_id"], claim) for claim in accepted]
                    except (LLMUnavailable, ValueError) as exc:
                        warnings.append({"code": "rewrite_fallback", "message":
                                         "英文改写未完成有效验证，本次保留已验证的原文引用。", "detail": str(exc)})
                backend = "llm"
                llm_verified_claims = len(accepted)
            except (LLMUnavailable, ValueError) as exc:
                if not self.allow_fallback:
                    raise
                warnings.append({"code": "llm_fallback",
                                 "message": "LLM 未完成有效生成，本次使用原文证据摘要。",
                                 "detail": str(exc)})
                backend = "extractive_fallback"
        # Supply one fact per product and preserve available opposing review
        # evidence even when the LLM selected only favorable statements.
        required = []
        for product in products:
            available = [f for f in facts if f.parent_asin == product["parent_asin"]]
            if available:
                required.append(available[0])
            for polarity in ("negative", "positive"):
                match = next((f for f in available if f.chunk_type == "review" and f.sentiment == polarity), None)
                if match is not None:
                    required.append(match)
        supplemental = required + facts
        # Mandatory evidence gets priority over optional LLM statements.
        required = list({f.fact_id:f for f in required}.values())[:self.max_claims]
        mandatory, _ = verify_claims({"claims": [{"fact_id": f.fact_id, "text": f.quote} for f in required]},
                                     facts, shown_evidence, max_claims=self.max_claims, english_only=True)
        # Preserve a verified single-fact rewrite for a required fact rather than duplicating it.
        selected_by_id = {claim["fact_id"]: claim for claim in accepted}
        mandatory_ids = {claim["fact_id"] for claim in mandatory}
        merged = [selected_by_id.get(claim["fact_id"], claim) for claim in mandatory]
        merged += [claim for claim in accepted if claim["fact_id"] not in mandatory_ids]
        if not accepted:
            fallback, _ = verify_claims({"claims": [{"fact_id": f.fact_id, "text": f.quote} for f in supplemental]},
                                        facts, shown_evidence, max_claims=self.max_claims, english_only=True)
            merged += fallback
            if backend == "llm":
                if not self.allow_fallback:
                    raise ValueError("LLM returned no supported claims.")
                backend = "extractive_fallback"
                warnings.append({"code": "no_supported_llm_claims",
                                 "message": "LLM 的结论未通过证据检查，本次使用原文证据摘要。"})
        accepted, identities = [], set()
        for claim in merged:
            identity = (claim["parent_asin"], claim["text"])
            if identity not in identities and len(accepted) < self.max_claims:
                identities.add(identity)
                accepted.append(claim)
        by_chunk = {item.chunk_id: item for item in shown_evidence}
        sources = {f.source_id: by_chunk[f.chunk_id].to_dict() for f in facts}
        for product in products:
            product["claims"] = [c for c in accepted if c["parent_asin"] == product["parent_asin"]]
            pcs, budget = product["pcs"], product["budget_check"]["status"]
            if not product["claims"] or pcs["score"] < 35:
                verdict = "insufficient_evidence"
            elif budget == "violated" or pcs["fit_score"] < 0.4:
                verdict = "not_preferred"
            elif budget == "unknown":
                verdict = "needs_budget_check"
            elif ("lexicon_baseline" in pcs["sentiment_methods"]
                  and not (product_signals or {}).get(product["parent_asin"], {}).get("aspects")):
                verdict = "needs_sentiment_validation"
            elif pcs["fit_score"] >= 0.6:
                verdict = "consider"
            else:
                verdict = "mixed_evidence"
            product["verdict"] = verdict
        finished = time.perf_counter()
        output = {"query": query, "requested_aspects": aspects, "constraints": constraints,
                  "product_filter": {'excluded_products':excluded_products, 'method':'conservative_title_rules',
                                     'feature_requirements_verified':False},
                  "prompt_version": self.prompt_version if self.llm is not None else None,
                  "products": products, "claims": accepted, "sources": sources,
                  "status": "ok" if accepted else "insufficient_evidence", "backend": backend,
                  "grounding": {"accepted": len(accepted), "rejected": rejected,
                                "llm_verified_claims": llm_verified_claims,
                                "llm_verified_selections": llm_selections,
                                "selected_fact_ids": sorted(selected_by_id),
                                "selection_rejected": selection_rejected,
                                "rewrite_verified_claims": rewrite_verified,
                                "mode": "nli_plus_id_selection" if self.verifier else "id_selection",
                                "conclusion_language": "en"},
                  "sentiments": {key: value.to_dict() for key, value in sentiments.items()},
                  "warnings": warnings,
                  "llm": self.llm.describe() if hasattr(self.llm, "describe") else None,
                  "llm_stages": {"selection": selection_info, "rewrite": rewrite_info},
                  "timing": {"preparation_seconds": prepared-started,
                             "generation_and_check_seconds": finished-prepared,
                             "total_seconds": finished-started}}
        output["answer"] = render_answer(output)
        return output


def display_text(text):
    """Clean the rendered view only; never change stored evidence, claims or spans."""
    text = unescape(text)
    text = re.sub(r'<[^>]*(?:>|$)', ' ', text)
    text = re.sub(r'(?:/>)', ' ', text)
    return escape(' '.join(text.split()))


def render_answer(output):
    if not output["claims"]:
        return "The available evidence is insufficient for a reliable purchase recommendation. Please provide relevant reviews or product information."
    labels = {"insufficient_evidence": "Insufficient evidence for a purchase recommendation",
              "not_preferred": "Not a preferred option", "needs_budget_check": "Budget requirements need verification",
              "needs_sentiment_validation": "Sentiment needs validation before making a purchase recommendation",
              "consider": "May be considered", "mixed_evidence": "Mixed evidence; compare with caution"}
    lines = []
    for product in output["products"]:
        pcs = product["pcs"]
        lines += ["## " + display_text(product["title"]),
                  "", f"{labels[product['verdict']]}. PCS: **{pcs['score']:.2f}/100**.", ""]
        for kind, title in (("finding", "Key evidence"), ("pro", "Positive reviews"), ("con", "Negative reviews")):
            subset = [c for c in product["claims"] if c["kind"] == kind]
            if subset:
                lines += ["### " + title, ""]
                for claim in subset:
                    refs = " ".join(f"[{'Review' if c['chunk_type']=='review' else 'Spec'} {c['source_id']}]"
                                    for c in claim["citations"])
                    text = display_text(claim["text"])
                    if claim['grounding_method'] == 'exact_quote' and claim['citations'][0]['chunk_type'] == 'review':
                        text = 'One reviewer wrote: “' + text + '”'
                    lines.append(f"- {text} {refs}")
                lines.append("")
        c = pcs["components"]
        lines += [f"{pcs['review_evidence_units']} deduplicated review evidence units; aspect coverage {c['coverage']:.0%}, "
                  f"polarization {c['polarization']:.0%}, evidence quality {c['quality']:.0%}.", ""]
        missing = pcs["missing_metadata"]
        if missing["timestamp"]:
            lines += ["Some reviews lack valid timestamps; recency uses a neutral default of 0.5.", ""]
        if missing["source_review_id"]:
            lines += ["Some original review IDs are missing; deduplicated text units are not counts of independent users.", ""]
        if "lexicon_baseline" in pcs["sentiment_methods"]:
            lines += ["Sentiment labels use an uncalibrated lexicon baseline; formal evaluation needs validated aspect sentiment.", ""]
        if missing["verified_purchase"] or missing["helpful_vote"]:
            lines += ["Some reviews lack verified-purchase or helpful-vote data; these fields add no evidence-quality credit.", ""]
        if product["budget_check"]["status"] == "unknown":
            lines += ["No verifiable price is available; the budget requirement remains unverified.", ""]
    lines += ["PCS is an experimental evidence-confidence metric based on the supplied sample.", "", "## Sources", ""]
    used_sources = {c["source_id"] for claim in output["claims"] for c in claim["citations"]}
    for source_id, source in output["sources"].items():
        if source_id in used_sources:
            lines.append(f"- {source_id}: `{source['chunk_id']}` ({source['chunk_type']}, {escape(source['aspect'])})")
    for warning in output["warnings"]:
        messages = {"llm_fallback": "The model could not complete valid selection; original evidence is shown.",
                    "no_supported_llm_claims": "No model selections passed validation; original evidence is shown.",
                    "rewrite_fallback": "English rewrites could not be validated; verified original quotations are retained."}
        lines += ["", messages.get(warning["code"], "See structured output for validation details.")]
    return "\n".join(lines).strip()
