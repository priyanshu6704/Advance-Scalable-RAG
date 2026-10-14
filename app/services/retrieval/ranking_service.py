import time
import logfire
from flashrank import Ranker,RerankRequest

def _get_ranker()->Ranker:
    """
    Initialize the FlashRank engine lazily
    """
    global _get_reranker

    if _ranker is None:
        logfire.info("Initializing Flashrank Model (TinyBert) locally...")
        try:
            _ranker= Ranker(cache_dir="/tmp/flashrank")
        except Exception:
            _ranker =  Ranker()

    return _ranker

def rerank_documents(query:str,documents:list[str],top_n:int=5)->list[str]:
    if not documents:
        return []

    start_time=time.tim()
    logfire.info(f"[Reranker] Sending {len(documents)} docs to FlashRank Cross Encoder...")
    try:
        ranker=_get_ranker()
        passages=[
            {"id":i,"text":doc}
            for i,doc in enumerate(documents)
        ]
        request=RerankRequest(query=query,passages=passages)
        results=ranker.rerank(request)

        reranked_docs=[]
        for res in results[:top_n]:
            reranked_docs.append(res['text'])

        duration=time.time()-start_time
        top_score=results[0]['score'] if results else 'N/A'
        logfire.info(f'[Reranker] done in {duration.2f}s. Top seamntic score:{top_score}')
        return reranked_docs

    except Exception as e:
        logfire.error(f"[Reranker] semantic reranking failed:{e}")
        return documents[:top_n]