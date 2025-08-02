from extract_features_cli import load_ground_truth, load_retriever_results
from src.retrievers.base import RetrievalResult
from src.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    evaluate_dataset,
    print_evaluation_results
)


results = load_retriever_results('./data/processed/results_3_retrievers.json')
results_xgb = load_retriever_results('./data/processed/results_xgb_val.json')

results.update(results_xgb)

gt = load_ground_truth('data/processed/ground_truth_final.json')


for name, retriever_results in results.items():
    if name == 'metadata':
        continue

    retriever_recall = {}

    for query, recall in retriever_results.items():
        retriever_recall[query] = [RetrievalResult(**r) for r in recall]

    metrics = evaluate_dataset(
        query_results=retriever_recall,
        ground_truth=gt,
        k_values=[3, 5, 10],
        threshold=2,
    )

    print(name)
    print(metrics)

    metrics = evaluate_dataset(
        query_results=retriever_recall,
        ground_truth=gt,
        k_values=[3, 5, 10],
        threshold=1,
    )
    print(metrics)
