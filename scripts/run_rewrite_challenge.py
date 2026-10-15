"""Small, explicitly developer-authored guard + real local NLI diagnostic."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.generation.evidence import normalize_evidence
from src.generation.grounding import Fact, NLIVerifier, verify_claims

CASES=[
 ('scope_loss','Battery life is twice what I need.','One reviewer reports that battery life is twice what is needed.',False),
 ('number_change','It lasts 5 hours.','One reviewer reports that it lasts 8 hours.',False),
 ('unit_loss','It lasts 5 hours.','One reviewer reports that it lasts 5.',False),
 ('negation_loss','It is not waterproof.','One reviewer reports that it is waterproof.',False),
 ('condition_loss','It works only if I restart it.','One reviewer reports that their device works.',False),
 ('uncertainty_loss','It may disconnect sometimes.','One reviewer reports that it disconnects.',False),
 ('time_loss','So far it works well.','One reviewer reports that it works well.',False),
 ('attribution_loss','The sound is clear.','The sound is clear and pleasant.',False),
 ('invented_capability','The sound is clear.','One reviewer reports that the speaker is waterproof.',False),
 ('plain_supported','The sound is clear.','One reviewer reports that the sound is clear.',True),
 ('numeric_supported','It lasts 5 hours.','One reviewer reports that it lasts 5 hours.',True),
 ('negative_supported','The connection does not work.','One reviewer reports that the connection does not work.',True),
 ('condition_supported','It works only if the phone stays nearby.','One reviewer reports that it works only if the phone stays nearby.',True),
 ('uncertain_supported','The connection sometimes drops.','One reviewer reports that the connection sometimes drops.',True),
 ('personal_supported','I get 5 hours of battery life.','One reviewer reports that they get 5 hours of battery life.',True),
]


def main():
    import torch
    torch.set_num_threads(4)
    verifier=NLIVerifier(threshold=0.9)
    outcomes=[]
    for name,source,claim,expected in CASES:
        items=normalize_evidence([{'parent_asin':'A','chunk_id':name,'title':'Diagnostic Device','text':source,'aspect':'general'}])
        fact=Fact('F1','E1',name,'A','review','general',source,0,len(source),'neutral',1.0)
        accepted,rejected=verify_claims({'claims':[{'fact_id':'F1','text':claim}]},[fact],items,verifier,english_only=True,strict_rewrite=True)
        outcomes.append({'case':name,'source':source,'claim':claim,'expected_supported':expected,'actual_supported':bool(accepted),
                         'matched_expectation':bool(accepted)==expected,'accepted':accepted,'rejected':rejected})
    report={'scope':'15 developer-authored diagnostic examples, not independent semantic accuracy or calibration.',
            'nli_model':verifier.model_name,'threshold':verifier.threshold,'matched':sum(r['matched_expectation'] for r in outcomes),
            'total':len(outcomes),'unsafe_accepted':sum(r['actual_supported'] and not r['expected_supported'] for r in outcomes),
            'supported_rejected':sum(not r['actual_supported'] and r['expected_supported'] for r in outcomes),'cases':outcomes}
    target=ROOT/'outputs/member_c_generation_testing_revised_final_2026-10-15/rewrite_challenge.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__': main()
