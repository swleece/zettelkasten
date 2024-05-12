---
date: 2024-05-12
time: 13:09
note_maturity: 🌱
tags:
  - idea
---
# Maximum Product Subarray




```Python
def maxProduct(self, nums: List[int]) -> int:
        """
        need to keep track of previous max subarray prod maximum
        need to keep track of sign pos/neg, can be done by keeping track of
          min subarray product
        dynamic programming
        """
        result = max(nums)
        current_max = 1
        current_min = 1

        for n in nums:
            if n == 0:
                current_max = 1
                current_min = 1
                continue
            
            tmp = current_max
            current_max = max(n * current_max, n * current_min, n)
            current_min = min(n * tmp, n * current_min, n)

            result = max(result, current_max)

        return result

```






#### 🧭  Idea Compass
- West  (similar) 
[[Computer Science]]
[[Algorithms]]
[[Maximum Sum Subarray Problem]]
- East (opposite)

- North (theme/question)

- South (what follows)
