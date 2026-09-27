import os

def new_matcher(predicted: str, target: str, threshold: float = 0.45) -> bool:
    """Semantic fuzzy matcher using a lightweight SentenceTransformer embedding."""
    pred_clean = predicted.lower().strip()
    target_clean = target.lower().strip()
    
    # Exact or substring match is an immediate pass
    if target_clean in pred_clean or pred_clean in target_clean:
        return True
        
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Load small fast model. Caching ensures we don't reload constantly.
        # Ensure we suppress warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        pred_emb = model.encode([pred_clean])
        target_emb = model.encode([target_clean])
        sim = cosine_similarity(pred_emb, target_emb)[0][0]
        
        # Specific known bad overlaps to reject forcefully even if similarity is high
        bad_overlaps = [
            ('water', 'built-up'),
            ('water', 'urban'),
            ('forest', 'desert'),
            ('vegetation', 'desert'),
            ('agriculture', 'urban'),
            ('building', 'water'),
            ('parking lot', 'water'),
            ('highway', 'water'),
            ('aircraft', 'tarmac'),
            ('solar', 'forest')
        ]
        
        for word1, word2 in bad_overlaps:
            if (word1 in pred_clean and word2 in target_clean) or (word2 in pred_clean and word1 in target_clean):
                return False
                
        return bool(sim >= threshold)
    except Exception as e:
        print(f"Warning: Semantic matching failed due to {e}. Falling back to strict matching.")
        return False
