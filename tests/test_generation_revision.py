import json
import unittest
from src.generation.generator import ShopSmartGenerator, display_text
from src.generation.grounding import rewrite_safety, canonical_product_name
from src.generation.product_match import requirements, filter_products
from src.generation.prompts import bounded_schema, build_selection_prompt, build_english_rewrite_prompt
from src.generation.evidence import normalize_evidence


class RevisionTests(unittest.TestCase):
    def test_configured_limit_reaches_both_transport_schemas_and_runtime(self):
        class Backend:
            def __init__(self): self.schemas = []
            def generate_structured(self, prompt, schema):
                self.schemas.append(schema)
                key = next(iter(schema['properties']))
                self_outer.assertEqual(schema['properties'][key]['maxItems'], 1)
                embedded = json.loads(prompt.split('Return only JSON conforming to this schema:')[1].splitlines()[1])
                self_outer.assertEqual(embedded, schema)
                if key == 'selections':
                    return {'selections':[{'fact_id':'F1','product_name':'Example Device'}]}
                return {'claims':[{'fact_id':'F1','text':'One reviewer reports that the connection failed.'}]}
        class Verifier:
            def supports(self, *args): return True, 0.99
        self_outer = self
        backend = Backend()
        result = ShopSmartGenerator(llm=backend, grounding_verifier=Verifier(), max_claims=1).generate_recommendation(
            'connection', [{'parent_asin':'A','chunk_id':'a','title':'Example Device','text':'The connection failed.','aspect':'connectivity'}])
        self.assertEqual(len(backend.schemas),2)
        self.assertEqual(len(result['claims']),1)

    def test_accessory_gate_applies_to_rule_supplements_and_all_excluded(self):
        rows = [{'parent_asin':p,'chunk_id':p,'title':title,'text':'Strong bass and good sound.','aspect':'sound quality'}
                for p,title in [('A','Travel Case for Portable Speaker'),('B','Portable Bluetooth Speaker')]]
        result = ShopSmartGenerator().generate_recommendation('Which portable speaker has strong bass?', rows)
        self.assertEqual([p['parent_asin'] for p in result['products']], ['B'])
        self.assertTrue(all(c['parent_asin']=='B' for c in result['claims']))
        empty = ShopSmartGenerator().generate_recommendation('Which portable speaker?', rows[:1])
        self.assertEqual(empty['status'],'insufficient_evidence')
        self.assertFalse(empty['claims'])

    def test_requested_accessories_and_device_bundles(self):
        for query,title in [('Find a protective tablet case','Case for iPad Tablet'),
                            ('Find a camera bag','Camera Backpack'),
                            ('Find a smartwatch','Amazfit Smartwatch 14-Day Battery Life'),
                            ('Find wireless earbuds','Wireless Earbuds with Charging Case')]:
            items = normalize_evidence([{'parent_asin':'A','chunk_id':'A','title':title,'text':'Good product.'}])
            kept,_ = filter_products(items, requirements(query,[],{}))
            self.assertEqual(len(kept),1,(query,title))

    def test_scope_numbers_negation_and_conditions_fail_before_nli(self):
        cases = [
            ('Battery life is twice what I need.','One reviewer reports that battery life is twice what is needed.','personal_scope_lost'),
            ('It lasts 5 hours.','One reviewer reports that it lasts 8 hours.','numeric_detail_changed'),
            ('It does not work.','One reviewer reports that it works.','negation_changed'),
            ('It works only if I restart it.','One reviewer reports that their device works.','condition_or_uncertainty_lost'),
            ('So far it may work.','One reviewer reports that it works.','condition_or_uncertainty_lost'),
            ('The sound is good.','The sound is good and clear.','review_attribution_missing'),
        ]
        for source,claim,reason in cases:
            self.assertEqual(rewrite_safety(source,claim),reason)
        self.assertIsNone(rewrite_safety('It lasts 5 hours for my needs.', 'One reviewer reports that it lasts 5 hours for their needs.'))

    def test_display_cleaning_does_not_change_original_offsets(self):
        original = 'Easy setup.<br /><br />-<br /><br'
        rows = [{'parent_asin':'A','chunk_id':'a','title':'Example','text':original,'aspect':'ease of use'}]
        result = ShopSmartGenerator().generate_recommendation('easy setup',rows)
        self.assertNotIn('&lt;br',result['answer'])
        for c in result['claims']:
            for citation in c['citations']:
                self.assertEqual(original[citation['start']:citation['end']],citation['quote'])
        self.assertEqual(display_text('A &amp; B<br/>C'),'A &amp; B C')

    def test_name_normalization_keeps_model_and_suffix(self):
        self.assertEqual(canonical_product_name(' A &amp; B  2024 '),'A & B 2024')
        self.assertNotEqual(canonical_product_name('A Pro 2024'),canonical_product_name('A Pro'))

    def test_nli_cannot_override_scope_guard(self):
        from src.generation.grounding import extract_facts, verify_claims
        from src.generation.sentiment import LexiconSentiment, evidence_sentiment
        items = normalize_evidence([{'parent_asin':'A','chunk_id':'a','title':'Example','text':'Battery life is twice what I need.','aspect':'battery life'}])
        classifier = LexiconSentiment()
        facts = extract_facts(items,'battery life',['battery life'],classifier,{'a':evidence_sentiment(items[0],classifier)})
        class Verifier:
            def supports(self,*args): raise AssertionError('Scope failure must precede NLI')
        accepted,rejected = verify_claims({'claims':[{'fact_id':'F1','text':'One reviewer reports that battery life is twice what is needed.'}]},facts,items,Verifier(),strict_rewrite=True)
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]['reason'],'personal_scope_lost')

    def test_nli_receives_proposition_after_attribution_guard(self):
        from src.generation.grounding import extract_facts, verify_claims
        from src.generation.sentiment import LexiconSentiment, evidence_sentiment
        items = normalize_evidence([{'parent_asin':'A','chunk_id':'a','title':'Example','text':'The connection failed.','aspect':'connectivity'}])
        classifier = LexiconSentiment()
        facts = extract_facts(items,'connection',['connectivity'],classifier,{'a':evidence_sentiment(items[0],classifier)})
        seen=[]
        class Verifier:
            def supports(self,premise,claim): seen.append(claim); return True,0.99
        accepted,_=verify_claims({'claims':[{'fact_id':'F1','text':'One reviewer reports that the connection failed.'}]},facts,items,Verifier(),strict_rewrite=True)
        self.assertEqual(seen,['the connection failed.'])
        self.assertEqual(accepted[0]['text'],'One reviewer reports that the connection failed.')


if __name__ == '__main__': unittest.main()
