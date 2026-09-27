import numpy as np

def test_matcher():
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Use a small fast model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    manual_check = [
        {'predicted': 'agricultural field', 'target': 'cultivated agriculture', 'should_match': True},
        {'predicted': 'water body', 'target': 'built-up area', 'should_match': False},
        {'predicted': 'vegetation', 'target': 'dense forest cover', 'should_match': True},
        {'predicted': 'urban infrastructure', 'target': 'built-up area', 'should_match': True},
        {'predicted': 'arid desert', 'target': 'dense forest cover', 'should_match': False},
        {'predicted': 'coastal water', 'target': 'water body', 'should_match': True},
        {'predicted': 'commercial aircraft', 'target': 'airport tarmac', 'should_match': False},
        {'predicted': 'residential neighborhood', 'target': 'urban', 'should_match': True},
        {'predicted': 'parking lot', 'target': 'water', 'should_match': False},
        {'predicted': 'bare soil', 'target': 'arid desert', 'should_match': True},
        {'predicted': 'commercial buildings', 'target': 'built-up', 'should_match': True},
        {'predicted': 'highway', 'target': 'road', 'should_match': True},
        {'predicted': 'highway', 'target': 'water', 'should_match': False},
        {'predicted': 'cultivated land', 'target': 'agriculture', 'should_match': True},
        {'predicted': 'solar panels', 'target': 'forest', 'should_match': False}
    ]
    
    for case in manual_check:
        pred_emb = model.encode([case['predicted']])
        target_emb = model.encode([case['target']])
        sim = cosine_similarity(pred_emb, target_emb)[0][0]
        
        print(f"[{case['should_match']}] {sim:.3f} | '{case['predicted']}' vs '{case['target']}'")

if __name__ == '__main__':
    test_matcher()
