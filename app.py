"""ShopSmart: evidence-led product discovery, comparison and research."""
import html
import json
import os
import re
import time
from pathlib import Path
import streamlit as st
from src.ui.backend import ShopSmartBackend

ROOT = Path(__file__).parent
st.set_page_config(page_title="ShopSmart | Buy with evidence", page_icon="S", layout="wide", initial_sidebar_state="expanded")
st.session_state.setdefault("theme", "Day")
st.session_state.setdefault("query", "")
st.session_state.setdefault("shortlist", {})
st.session_state.setdefault("history", [])
night = st.session_state.theme == "Night"
palette = dict(bg="#101917", surface="#182420", soft="#21332b", ink="#edf5ef", muted="#b3c5ba", line="#3e5548", accent="#b9dfb0", onaccent="#182b20") if night else dict(bg="#f6f7f2", surface="#ffffff", soft="#eaf0e5", ink="#1d342b", muted="#53675c", line="#d5dfd3", accent="#235840", onaccent="#ffffff")
variables = ':root{' + ';'.join(f'--{k}:{v}' for k,v in palette.items()) + ';}'
st.markdown('<style>'+variables+(ROOT/'ui.css').read_text(encoding='utf-8')+'</style>', unsafe_allow_html=True)

def markup(text):
    st.markdown(text, unsafe_allow_html=True)

def esc(text):
    return html.escape(str(text))

def heading(kicker, title, description=None):
    markup(f'<div class="eyebrow">{esc(kicker)}</div>')
    markup(f'<h1>{esc(title).replace(chr(10), "<br>")}</h1>')
    if description:
        st.caption(description)

MODE = os.getenv('SHOPSMART_MODE') or ('full' if Path(os.getenv('CHUNKS_PATH','chunks.parquet')).exists() else 'saved')
@st.cache_resource
def load_backend(mode):
    if mode == 'saved':
        return ShopSmartBackend.from_saved_results()
    return ShopSmartBackend.load(chunks_path=os.getenv('CHUNKS_PATH','chunks.parquet'), chroma_path=os.getenv('CHROMA_PATH','chroma_db'), title_cache=os.getenv('TITLE_CACHE','title_vectors.npz'), openai_key=os.getenv('OPENAI_API_KEY'))
try:
    with st.spinner('Preparing your shopping desk...'):
        backend = load_backend(MODE)
except Exception as exc:
    st.error('We could not load the product information. Please try again shortly.')
    st.stop()
questions = backend.saved_questions() if MODE == 'saved' else [json.loads(x)['query'] for x in (ROOT/'evaluation/formal_queries_v1.jsonl').read_text().splitlines()]

def choose(q):
    st.session_state.query = q
    st.session_state.search_input = q
    st.session_state.page = 'Discover'

def save_product(hit, analysis):
    st.session_state.shortlist[hit.asin] = dict(hit=hit, analysis=analysis, query=st.session_state.query)

def risk_panel(hit, query):
    result = backend.explain(hit.asin, query, with_alternatives=False)
    st.write(result.get('headline', 'Review evidence is limited.'))
    for group, label in [('deal_breakers','Deal breaker'),('complaints','Common complaint'),('polarized','Mixed opinions'),('minor_concerns','Minor concern')]:
        for item in result.get(group, []):
            st.markdown(f"**{label} · {item.get('issue', item.get('aspect', 'Review finding'))}**")
            evidence = item.get('evidence', [])
            if isinstance(evidence, dict): evidence = [evidence]
            for e in evidence:
                st.write(e.get('quote',''))
                st.caption(e.get('chunk_id',''))
            if group == 'polarized':
                st.caption(f"{item.get('positive_share',0):.0%} positive / {item.get('negative_share',0):.0%} negative")
            for alt in item.get('alternatives',[]):
                st.write(f"Alternative: {alt.get('title') or alt['asin']}")
    if result.get('deal_breakers'):
        if st.button('Find alternatives', key=f'alternatives_{hit.asin}', width='stretch'):
            with st.spinner('Checking alternatives against the reviews...'):
                alternatives = backend.explain(hit.asin, query, with_alternatives=True)
                for issue in alternatives.get('deal_breakers', []):
                    for alt in issue.get('alternatives', []):
                        st.write(alt.get('title') or alt['asin'])
    for missing in result.get('missing',[]):
        st.caption(f"Limited evidence · {missing['aspect']} ({missing.get('chunks',0)} passages)")

with st.sidebar:
    markup('<div class="brand"><span class="brand-symbol">S</span> ShopSmart<span class="brand-dot">.</span></div>')
    markup('<div class="nav-caption">THE SHOPPING RESEARCH DESK</div>')
    if st.session_state.get('page') not in ('Discover', 'Compare'):
        st.session_state.page = 'Discover'
    page = st.radio('Workspace', ['Discover', 'Compare'], label_visibility='collapsed', key='page')
    st.divider()
    st.caption('APPEARANCE')
    st.radio('Appearance', ['Day','Night'], key='theme', horizontal=True, label_visibility='collapsed')
    if page == 'Discover':
        st.divider()
        n_products = st.select_slider('Results per search', [3,5,8], value=3)
    st.divider()
    if st.session_state.query:
        st.button('New search',on_click=choose,args=('',),width='stretch')

markup('<div class="topbar"><span>ELECTRONICS / BUYING INTELLIGENCE</span><span>Evidence over opinion</span></div>')
if page == 'Compare':
    heading('YOUR SHORTLIST', 'The details, side by side.', 'Compare evidence, requirement coverage and trade-offs before you decide.')
    saved=st.session_state.shortlist
    if not saved:
        with st.container(border=True):
            st.subheader('Your next purchase deserves a closer look.')
            st.write('Add up to three products from Discover to compare them here.')
            st.button('Explore products', on_click=lambda: st.session_state.update(page='Discover'), type='primary')
    else:
        cols=st.columns(len(saved))
        for col,(asin,item) in zip(cols,list(saved.items())):
            hit,analysis=item['hit'],item['analysis']
            with col,st.container(border=True,key=f'card_compare_{asin}'):
                st.caption(hit.category or 'Electronics')
                st.markdown(f'#### {hit.title}')
                st.caption(f'ASIN {asin}')
                st.metric('Purchase confidence',f"{analysis['pcs']['score']}/100")
                st.caption('Based on available review evidence.')
                st.markdown('**Requirement coverage**')
                for c in analysis['coverage']:
                    st.write(f"{c['aspect']} — {dict(covered='Supported',partial='Limited',missing='No evidence')[c['status']]}")
                st.markdown('**Review finding**')
                st.write(analysis['why_not'].get('headline','Insufficient review evidence.'))
                with st.expander('Evidence'):
                    for e in hit.evidence:
                        st.write(e.quote); st.caption(e.chunk_id)
                st.caption(f"Search: {item['query']}")
                st.button('Remove product',key=f'remove_{asin}',on_click=lambda a=asin:st.session_state.shortlist.pop(a,None),width='stretch')
        export=[dict(asin=a,title=v['hit'].title,query=v['query'],pcs=v['analysis']['pcs'],coverage=v['analysis']['coverage'],evidence=[vars(e) for e in v['hit'].evidence]) for a,v in saved.items()]
        st.download_button('Export comparison',json.dumps(export,indent=2), 'shopsmart-comparison.json','application/json')
else:
    if not st.session_state.query:
        left,right=st.columns([1.5,1],gap='large')
        with left:
            heading('LESS GUESSWORK. BETTER CHOICES.', 'Find your fit.\nKnow the trade-offs.')
            st.write('Shopping advice grounded in real product reviews. See what fits, what falls short, and why.')
        with right:
            markup('<div class="editorial"><span class="editorial-label">THE SHOPSMART DIFFERENCE</span><div class="editorial-title">More than a<br>star rating.</div><div class="editorial-row"><b>01</b> Evidence you can inspect</div><div class="editorial-row"><b>02</b> Coverage of your requirements</div><div class="editorial-row"><b>03</b> A reason to think twice</div></div>')
    else:
        heading('YOUR BUYING BRIEF', 'Good choices have a reason.')
    with st.form('search'):
        st.text_input('What are you looking for?',key='search_input',placeholder='Wireless earbuds for long flights, with good battery life')
        submitted = st.form_submit_button('Find my options',type='primary',width='stretch')
    if submitted:
        st.session_state.query=st.session_state.search_input.strip()
        if st.session_state.query: st.rerun()
        if not st.session_state.query: st.warning('Enter a product or a requirement to start.')
    if not st.session_state.query:
        st.subheader('Start with what matters to you')
        cols=st.columns(3)
        labels=['Long-haul listening','Clearer work calls','Sound that travels']
        notes=['Comfort & battery life','Microphone & connection','Bass & volume']
        for i,col in enumerate(cols):
            with col,st.container(border=True,key=f'card_example_{i}'):
                markup(f'<div class="step">0{i+1} / AUDIO</div>')
                st.markdown(f'### {labels[i]}')
                st.caption(notes[i])
                st.button('Explore options',key=f'example_{i}',on_click=choose,args=(questions[i],),width='stretch')
        with st.expander('More shopping ideas'):
            q=st.selectbox('What matters to you?',['']+questions,format_func=lambda x:x or 'Select a question')
            st.button('Explore these options',disabled=not q,on_click=choose,args=(q,),width='stretch')
    else:
        query=st.session_state.query
        st.caption(f'SEARCH BRIEF / {query}')
        search_key = (MODE, query, n_products)
        if st.session_state.get('last_search_key') != search_key:
            with st.spinner('Finding relevant product reviews...'):
                t0=time.perf_counter()
                hits,_=backend.search(query,n_products)
                elapsed=time.perf_counter()-t0
                st.session_state.last_search = (hits, elapsed)
                st.session_state.last_search_key = search_key
        hits, elapsed = st.session_state.last_search
        if not st.session_state.history or st.session_state.history[-1]['Query']!=query:
            st.session_state.history.append({'Query':query,'Products':len(hits),'Search (s)':round(elapsed,4),'Mode':MODE})
        if backend.is_approximate(query):
            st.caption('Results are based on a limited set of available reviews.')
        if not hits:
            st.info('No matching evidence. Try a more specific product or requirement.')
        else:
            with st.container(border=True,key='card_recommendation'):
                markup('<div class="eyebrow">THE RECOMMENDATION</div>')
                if st.session_state.get('last_answer_key') != search_key:
                    with st.spinner('Putting the evidence together...'):
                        st.session_state.last_answer = backend.answer(query,hits)
                        st.session_state.last_answer_key = search_key
                answer = st.session_state.last_answer
                answer_text = re.sub(r'^\(LLM unavailable:[^\n]*\)\s*', 'Here is the available review evidence.\n\n', answer['text'])
                answer_text = answer_text.replace('Generated from review quotes only (no language model connected).', 'Review highlights with linked sources.')
                st.markdown(answer_text)
                if answer['citations']:
                    with st.expander('Inspect cited sources'):
                        for cid,c in answer['citations'].items():
                            st.markdown(f"**[{cid}] {c['product']}**")
                            st.write(c['quote']);st.caption(c['chunk_id'])
            st.subheader(f'{len(hits)} options for your shortlist')
            st.caption('Ranked by retrieval relevance. Confidence reflects the available evidence.')
            for hit in hits:
                with st.spinner('Checking product fit and review evidence...'):
                    analysis=backend.analyse(hit,query)
                pcs=analysis['pcs']
                with st.container(border=True,key=f'card_product_{hit.asin}'):
                    left,right=st.columns([3,1],gap='large')
                    with left:
                        st.caption(f'OPTION {hit.rank:02d} / {hit.category or "Electronics"}')
                        st.markdown(f'### {hit.title}')
                        st.caption(f'ASIN {hit.asin}' + (f' · Review rating {hit.avg_rating:.1f}/5' if hit.avg_rating else ''))
                    with right:
                        st.metric('Purchase confidence',f"{pcs['score']}/100")
                        st.caption('Based on available review evidence.')
                    tabs=st.tabs(['Evidence','Requirement fit','Why not?'])
                    with tabs[0]:
                        for e in hit.evidence:
                            st.write(e.quote)
                            st.caption(f'{e.aspect} · {e.chunk_id}')
                    with tabs[1]:
                        for c in analysis['coverage']:
                            markup(f'<div class="coverage-row"><span>{esc(c["aspect"])}</span><b class="status-{c["status"]}">{dict(covered="Supported",partial="Limited evidence",missing="No evidence")[c["status"]]}</b></div>')
                            if c.get('note'):st.caption(c['note'])
                        if not analysis['coverage']:st.caption('Add specific requirements to your query to check coverage.')
                        with st.expander('Score breakdown'):
                            for name,value in pcs['components'].items():
                                st.write(f"{name.replace('_',' ').capitalize()} · {value:.2f}")
                            st.caption(pcs['explanation'])
                    with tabs[2]:risk_panel(hit,query)
                    added=hit.asin in st.session_state.shortlist
                    full=len(st.session_state.shortlist)>=3
                    st.button('Added to comparison' if added else 'Add to comparison',key=f'add_{hit.asin}',disabled=added or full,on_click=save_product,args=(hit,analysis),width='stretch')
            if len(st.session_state.shortlist)>=3:st.caption('Comparison holds three products. Remove one in Compare to add another.')
markup('<div class="footer"><b>ShopSmart.</b><span>Informed choices. Traceable evidence.</span><span>Electronics, considered.</span></div>')
