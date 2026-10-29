# RAGAS and efficiency results

Generator `gpt-4o-mini`, judge `gpt-4o-mini`, k=5, 20 queries.

## RAGAS (mean over queries)

|                                      |   mean |   std |   min |   max |
|:-------------------------------------|-------:|------:|------:|------:|
| faithfulness                         |  0.702 | 0.209 | 0.333 | 1     |
| answer_relevancy                     |  0.715 | 0.32  | 0     | 0.935 |
| llm_context_precision_with_reference |  0.891 | 0.209 | 0.2   | 1     |
| context_recall                       |  0.46  | 0.264 | 0.091 | 0.85  |

## Efficiency

|                                       |        0 |
|:--------------------------------------|---------:|
| queries                               |  20      |
| retrieval_s_mean                      |   0.3947 |
| generation_s_mean                     |   2.9233 |
| generation_s_p95                      |   3.5236 |
| end_to_end_s_mean                     |   3.3181 |
| throughput_queries_per_min_sequential |  18.0827 |
| tokens_in_mean                        | 576      |
| tokens_out_mean                       | 257      |
| cost_per_query_usd                    |   0.0002 |
| cost_total_usd                        |   0.0048 |

Notes: context recall uses the human-validated grade-2 chunks as the reference (a proxy for a gold answer). Retrieval latency comes from the project's formal retrieval log. Throughput is sequential, single-process.
