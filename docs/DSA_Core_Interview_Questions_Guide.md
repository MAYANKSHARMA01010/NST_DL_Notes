# Core Data Structures & Algorithms: Teacher's Priority Problem Set

> **Course**: Deep Learning & Advanced Problem Solving  
> **Topic**: Foundational Interview Patterns & Data Structure Mastery  
> **Target Problems**: Two Sum, Best Time to Buy and Sell Stock, Valid Parentheses, Merge Intervals, Number of Islands  

---

## 📌 Executive Summary & Pattern Taxonomy

The five problems selected by your instructor represent the **foundational pillars of technical interviews and algorithm design**. Mastering these problems teaches you how to recognize core patterns that unlock hundreds of advanced interview questions:

| # | Problem | LeetCode | Difficulty | Core Data Structure | Algorithmic Pattern | Time | Space |
| :-: | :--- | :---: | :---: | :--- | :--- | :---: | :---: |
| **1** | [Two Sum](#1-two-sum) | [#1](https://leetcode.com/problems/two-sum/) | Easy | Hash Map (`dict`) | Complement Lookup / Hashing | $O(n)$ | $O(n)$ |
| **2** | [Best Time to Buy and Sell Stock](#2-best-time-to-buy-and-sell-stock) | [#121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) | Easy | Primitive Variables | One-Pass Running Extrema / Sliding Window | $O(n)$ | $O(1)$ |
| **3** | [Valid Parentheses](#3-valid-parentheses) | [#20](https://leetcode.com/problems/valid-parentheses/) | Easy | Stack (`list`) | LIFO Invariant Matching | $O(n)$ | $O(n)$ |
| **4** | [Merge Intervals](#4-merge-intervals) | [#56](https://leetcode.com/problems/merge-intervals/) | Medium | Array (`list`) | Sorting + Greedy Interval Consolidation | $O(n \log n)$ | $O(n)$ |
| **5** | [Number of Islands](#5-number-of-islands) | [#200](https://leetcode.com/problems/number-of-islands/) | Medium | 2D Matrix Grid | Graph Traversal (DFS / BFS Flood Fill) | $O(m \times n)$ | $O(m \times n)$ |

---

## 1. Two Sum

### 📖 Problem Statement
Given an array of integers `nums` and an integer `target`, return the **indices** of the two numbers such that they add up to `target`.
- You may assume that each input would have **exactly one solution**.
- You may not use the same element twice.
- You can return the answer in any order.

* **LeetCode Link**: [https://leetcode.com/problems/two-sum/](https://leetcode.com/problems/two-sum/)

---

### 💡 Core Intuition & Pattern: The Complement Hash Map
The brute-force solution checks every pair $(i, j)$ using nested loops, taking $O(n^2)$ time.

To achieve $O(n)$ time, flip the question:
> "Instead of asking *what number adds with $x$ to equal target?*, ask: **Have I already seen $(target - x)$ earlier in the array?**"

By storing elements we have already inspected in a **Hash Map** (`num_to_index`), lookup takes $O(1)$ average time.

```text
Array:  [2,  7, 11, 15],  Target = 9

Step 1: num = 2  -> complement = 9 - 2 = 7. Seen 7? No.  Store {2: 0}
Step 2: num = 7  -> complement = 9 - 7 = 2. Seen 2? Yes! At index 0.
        ==> Return [0, 1]!
```

---

### 💻 Production-Quality Python Solution

```python
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        Finds indices of the two numbers that add up to target.
        
        Time Complexity:  O(n) - Single pass through the array.
        Space Complexity: O(n) - Hash map stores at most n - 1 elements.
        """
        seen_complements = {}  # Map: value -> index
        
        for current_idx, num in enumerate(nums):
            complement = target - num
            
            # Check if the needed complement was already visited
            if complement in seen_complements:
                return [seen_complements[complement], current_idx]
            
            # Record current number and its index
            seen_complements[num] = current_idx
            
        return []  # Fallback if no solution exists
```

---

### 🔬 Step-by-Step Execution Trace

**Input**: `nums = [3, 2, 4]`, `target = 6`

| Iteration | `current_idx` | `num` | `complement = 6 - num` | `complement in seen?` | `seen_complements` State | Action |
| :-: | :-: | :-: | :-: | :-: | :--- | :--- |
| **0** | 0 | 3 | 3 | False (empty) | `{3: 0}` | Store 3 |
| **1** | 1 | 2 | 4 | False | `{3: 0, 2: 1}` | Store 2 |
| **2** | 2 | 4 | 2 | **True** (at index 1) | `{3: 0, 2: 1}` | **Return `[1, 2]`** |

---

### ⚠️ Edge Cases & Interview Traps
1. **Duplicate Elements**: e.g., `nums = [3, 3], target = 6`. The single-pass approach checks the complement *before* adding the current duplicate to the map, correctly pairing index 0 with index 1 without key collisions.
2. **Negative Numbers**: e.g., `nums = [-1, -3, 4], target = 3`. Complement arithmetic `3 - (-1) = 4` works seamlessly with standard integer algebra.
3. **Large Arrays**: Using `dict` provides $O(1)$ average amortization.

---

## 2. Best Time to Buy and Sell Stock

### 📖 Problem Statement
You are given an array `prices` where `prices[i]` is the price of a given stock on the $i$-th day.
You want to maximize your profit by choosing a **single day to buy** one stock and choosing a **different day in the future to sell** that stock.
Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return `0`.

* **LeetCode Link**: [https://leetcode.com/problems/best-time-to-buy-and-sell-stock/](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

---

### 💡 Core Intuition: One-Pass Running Minimum (Greedy / Sliding Window)
You can only sell **after** you buy. As you iterate forward in time:
1. Maintain the **lowest price seen so far** (`min_price`).
2. At every day $i$, calculate potential profit: `current_price - min_price`.
3. Update `max_profit` if this potential profit exceeds our record.

```text
Prices: [7, 1, 5, 3, 6, 4]

Day 0: Price = 7 | min_price = 7 | profit = 0
Day 1: Price = 1 | min_price = 1 | profit = 0
Day 2: Price = 5 | min_price = 1 | profit = 5 - 1 = 4
Day 3: Price = 3 | min_price = 1 | profit = 3 - 1 = 2 (max remains 4)
Day 4: Price = 6 | min_price = 1 | profit = 6 - 1 = 5 (New Max Profit: 5!)
Day 5: Price = 4 | min_price = 1 | profit = 4 - 1 = 3 (max remains 5)

==> Result: 5 (Buy on Day 1 at ₹1, Sell on Day 4 at ₹6)
```

---

### 💻 Production-Quality Python Solution

```python
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """
        Calculates maximum single-transaction profit.
        
        Time Complexity:  O(n) - Single forward pass through prices.
        Space Complexity: O(1) - Constant auxiliary storage.
        """
        if not prices or len(prices) < 2:
            return 0
        
        min_price = float('inf')
        max_profit = 0
        
        for price in prices:
            # Update the lowest purchase price observed so far
            if price < min_price:
                min_price = price
            # Or evaluate if selling today yields a higher profit
            elif price - min_price > max_profit:
                max_profit = price - min_price
                
        return max_profit
```

---

### 🔬 Complexity Analysis
- **Time Complexity**: $\mathcal{O}(n)$ — We scan through the `prices` array once.
- **Space Complexity**: $\mathcal{O}(1)$ — Only two scalar variables (`min_price`, `max_profit`) are tracked.

---

### ⚠️ Edge Cases & Interview Traps
1. **Monotonically Decreasing Prices**: e.g., `[7, 6, 4, 3, 1]`. The minimum price keeps updating downward, but potential profit is never $> 0$. Correctly returns `0`.
2. **Single Element or Empty Array**: `[5]` or `[]`. Cannot execute buy and sell on separate days; correctly returns `0`.
3. **Identical Prices**: `[4, 4, 4, 4]`. Returns `0`.

---

## 3. Valid Parentheses

### 📖 Problem Statement
Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.
An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

* **LeetCode Link**: [https://leetcode.com/problems/valid-parentheses/](https://leetcode.com/problems/valid-parentheses/)

---

### 💡 Core Intuition: Stack (Last-In, First-Out - LIFO)
Parentheses nesting follows a strict **LIFO (Last-In, First-Out)** invariant:
> *The most recently opened bracket must be the first one to be closed.*

When encountering an opening bracket `(`, `{`, `[`, push it onto a stack. When encountering a closing bracket `)`, `}`, `]`, the top element of the stack **must** be its matching opener.

```text
Input: "({[]})"

Read '(': Push '('   -> Stack: ['(']
Read '{': Push '{'   -> Stack: ['(', '{']
Read '[': Push '['   -> Stack: ['(', '{', '[']
Read ']': Match '['? YES! Pop '[' -> Stack: ['(', '{']
Read '}': Match '{'? YES! Pop '{' -> Stack: ['(']
Read ')': Match '('? YES! Pop '(' -> Stack: []
End of string and stack is EMPTY -> VALID (True)!
```

---

### 💻 Production-Quality Python Solution

```python
class Solution:
    def isValid(self, s: str) -> bool:
        """
        Validates bracket nesting order using a LIFO stack.
        
        Time Complexity:  O(n) - Single pass through string characters.
        Space Complexity: O(n) - Stack stores at most n opening brackets.
        """
        # An odd-length string cannot possibly be balanced
        if len(s) % 2 != 0:
            return False
            
        # Map each closing bracket to its required opening bracket
        matching_opener = {
            ')': '(',
            '}': '{',
            ']': '['
        }
        stack = []
        
        for char in s:
            if char in matching_opener:
                # If char is a closing bracket, pop from stack or use sentinel if empty
                top_element = stack.pop() if stack else '#'
                if top_element != matching_opener[char]:
                    return False
            else:
                # char is an opening bracket; push onto stack
                stack.append(char)
                
        # String is valid only if all opened brackets were properly matched and closed
        return len(stack) == 0
```

---

### 🔬 Complexity Analysis
- **Time Complexity**: $\mathcal{O}(n)$ — Traversing string of length $n$ with $O(1)$ push/pop operations.
- **Space Complexity**: $\mathcal{O}(n)$ — In the worst case (e.g., `"(((((("`), the stack holds $n$ elements.

---

### ⚠️ Edge Cases & Interview Traps
1. **Closing Bracket First**: e.g., `"]"` or `")("`. Attempting to pop an empty stack must be safely handled without raising `IndexError`.
2. **Leftover Openers**: e.g., `"(()"`. The loop finishes with elements still in the stack. Checking `len(stack) == 0` catches this.
3. **Mismatched Interleaving**: e.g., `"([)]"`. While count of openers and closers match, the nesting order is violated.

---

## 4. Merge Intervals

### 📖 Problem Statement
Given an array of `intervals` where `intervals[i] = [start_i, end_i]`, merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

* **LeetCode Link**: [https://leetcode.com/problems/merge-intervals/](https://leetcode.com/problems/merge-intervals/)

---

### 💡 Core Intuition: Sort by Start Time + Greedy Consolidation
If intervals are scattered arbitrarily, checking for overlaps requires comparisons across all pairs ($O(n^2)$).
However, if we **sort intervals by their starting times**:
1. Any intervals that can overlap must appear **adjacent** to each other in the sorted list.
2. For each interval $[start, end]$:
   - If $start \le \text{last merged interval's end}$, they overlap! Extend the existing interval's end: $\text{new\_end} = \max(\text{last\_end}, end)$.
   - If $start > \text{last merged interval's end}$, there is a gap. Start a new interval in our output list.

```text
Intervals: [[1, 3], [8, 10], [2, 6], [15, 18]]

1. Sort by start: [[1, 3], [2, 6], [8, 10], [15, 18]]

2. Step-by-step:
   Start merged = [[1, 3]]
   - Check [2, 6]:  2 <= 3 (Overlap!) -> merged[-1] becomes [1, max(3, 6)] = [1, 6]
   - Check [8, 10]: 8 > 6  (No overlap) -> append [8, 10]
   - Check [15, 18]: 15 > 10 (No overlap) -> append [15, 18]

==> Output: [[1, 6], [8, 10], [15, 18]]
```

---

### 💻 Production-Quality Python Solution

```python
from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        Sorts intervals by start time and consolidates overlapping segments.
        
        Time Complexity:  O(n log n) - Dominated by sorting.
        Space Complexity: O(n) - Output merged list (or O(log n) for sort recursion).
        """
        if not intervals:
            return []
            
        # Step 1: Sort in-place by interval start time
        intervals.sort(key=lambda x: x[0])
        
        merged = [intervals[0]]
        
        # Step 2: Iterate through sorted intervals and merge when overlap occurs
        for current_start, current_end in intervals[1:]:
            last_merged_end = merged[-1][1]
            
            if current_start <= last_merged_end:
                # Overlap detected: expand the current interval's upper bound
                merged[-1][1] = max(last_merged_end, current_end)
            else:
                # Disjoint interval: append directly
                merged.append([current_start, current_end])
                
        return merged
```

---

### 🔬 Complexity Analysis
- **Time Complexity**: $\mathcal{O}(n \log n)$ — Timsort on $n$ intervals dominates the linear $\mathcal{O}(n)$ sweep.
- **Space Complexity**: $\mathcal{O}(n)$ — For storing the output intervals (or $\mathcal{O}(\log n)$ auxiliary space utilized by Python's sorting).

---

### ⚠️ Edge Cases & Interview Traps
1. **Fully Enclosed Intervals**: e.g., `[[1, 10], [2, 6]]`. The end of `[2, 6]` is smaller than 10. `max(last_end, current_end)` ensures we do not shrink the boundary from 10 to 6.
2. **Adjacent Touching Boundaries**: e.g., `[[1, 4], [4, 5]]`. Touching boundaries ($start = end$) count as overlapping; correctly merged into `[[1, 5]]`.
3. **Already Disjoint**: e.g., `[[1, 2], [3, 4]]`. Correctly preserves both intervals unchanged.

---

## 5. Number of Islands

### 📖 Problem Statement
Given an $m \times n$ 2D binary grid `grid` which represents a map of `'1'`s (land) and `'0'`s (water), return the **number of islands**.
An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

* **LeetCode Link**: [https://leetcode.com/problems/number-of-islands/](https://leetcode.com/problems/number-of-islands/)

---

### 💡 Core Intuition: Graph Connected Components & Flood Fill (DFS/BFS)
Think of the 2D grid as an **unconnected graph** where each `'1'` is a vertex connected to its 4 orthogonal neighbors (Up, Down, Left, Right).

1. Scan through every cell $(r, c)$ in the matrix.
2. When we land on a cell containing `'1'` (unvisited land):
   - We have discovered a **new island** $\rightarrow$ increment `island_count += 1`.
   - Launch a **Depth-First Search (DFS)** from this cell to "sink" the entire island (turn all connected `'1'`s into `'0'`s).
3. Sinking the island prevents re-counting cells of the same landmass and eliminates the need for an external $O(m \times n)$ visited set.

```text
Grid:
["1", "1", "0", "0", "0"],
["1", "1", "0", "0", "0"],
["0", "0", "1", "0", "0"],
["0", "0", "0", "1", "1"]

1. At (0,0): Land found! count = 1.
   DFS sinks (0,0), (0,1), (1,0), (1,1) to '0'.
2. At (2,2): Land found! count = 2.
   DFS sinks (2,2) to '0'.
3. At (3,3): Land found! count = 3.
   DFS sinks (3,3), (3,4) to '0'.

==> Total Islands = 3
```

---

### 💻 Production-Quality Python Solution (In-Place DFS)

```python
from typing import List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        Counts connected components of '1's in an m x n 2D grid using DFS flood-fill.
        
        Time Complexity:  O(m * n) - Each cell is visited a constant number of times.
        Space Complexity: O(m * n) - Recursion stack in the worst case (all land).
        """
        if not grid or not grid[0]:
            return 0
            
        rows, cols = len(grid), len(grid[0])
        island_count = 0
        
        def dfs(r: int, c: int) -> None:
            # Boundary checks and water/visited check
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
                return
                
            # Sink the current land cell to avoid re-visiting
            grid[r][c] = '0'
            
            # Explore all 4 orthogonal directions: Down, Up, Right, Left
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)
            
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    island_count += 1
                    dfs(r, c)  # Flood fill to sink all connected land
                    
        return island_count
```

---

### 🔬 Alternative: Breadth-First Search (BFS) Implementation
For deeply nested or massive grids where recursion depth exceeds Python's stack limits, an iterative BFS using `collections.deque` is the industry standard:

```python
from collections import deque
from typing import List

class SolutionBFS:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0
            
        rows, cols = len(grid), len(grid[0])
        island_count = 0
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    island_count += 1
                    grid[r][c] = '0'
                    queue = deque([(r, c)])
                    
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                                grid[nr][nc] = '0'
                                queue.append((nr, nc))
                                
        return island_count
```

---

### 🔬 Complexity Analysis
- **Time Complexity**: $\mathcal{O}(m \times n)$ — Every cell is visited at most once during the outer loop and once during the DFS/BFS traversal.
- **Space Complexity**: $\mathcal{O}(m \times n)$ — In the worst-case scenario (e.g., entire grid is land `'1'`), the call stack or queue will hold up to $m \times n$ frames.

---

### ⚠️ Edge Cases & Interview Traps
1. **Diagonal Land**: The problem specifies that lands connect only *horizontally or vertically*. Diagonal neighbors do **not** form a single island:
   ```text
   ["1", "0"]
   ["0", "1"]  --> 2 separate islands!
   ```
2. **All Water / All Land**: A grid of all `'0'`s outputs `0`. A grid of all `'1'`s outputs `1`.
3. **Modifying Input In-Place vs. Visited Set**: If the interviewer asks: *"What if the input matrix is immutable or read-only?"*, use an external `visited = set()` of `(r, c)` tuples to avoid mutating `grid`.

---

## 🧠 Master Cheat Sheet: Pattern Recognition in Interviews

When reading an interview question, look for these trigger phrases:

```text
"Find pair with target sum..."             ==> Hash Map Complement (Two Sum pattern)
"Find max profit with single buy/sell..." ==> One-Pass Running Minimum (Stock pattern)
"Check if braces/tags match properly..." ==> Stack LIFO Tracking (Valid Parentheses pattern)
"Merge, insert, or overlap ranges..."     ==> Sort by Start Time (Merge Intervals pattern)
"Count connected clusters / flood fill..."==> 2D Matrix DFS / BFS (Number of Islands pattern)
```
