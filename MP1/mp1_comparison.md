# MP1 Comparison Table

| strategy   |   Accuracy (mean of 3) |   Parse rate |   Total cost ($) |   Latency p50 (s) |
|:-----------|-----------------------:|-------------:|-----------------:|------------------:|
| cot        |                    2.7 |            1 |            0.001 |             1.398 |
| few_shot   |                    2.9 |            1 |            0.001 |             0.871 |
| structured |                    2.7 |            1 |            0.001 |             1.064 |
| zero_shot  |                    2.6 |            1 |            0     |             1.028 |

## Notes
- Accuracy: 0-3 scale (count of correct fields)
- Cost: Total USD for all 10 snippets
- Latency: Median time per call (seconds)
- Parse Success: % of responses that parsed as valid JSON
