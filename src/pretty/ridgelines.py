from rich.console import Console
from rich.text import Text
from typing import List, Dict
from rich.color import ANSI_COLOR_NAMES
from typing import List, Dict, Tuple
from collections import Counter

COLORS = [c for c in ANSI_COLOR_NAMES if "bright" in c or c in ["purple", "orange3", "turquoise2", "deep_pink2"]][1:]

console = Console()

def ridgeline_plot(
    distributions: List[Dict[str, float]], 
    x_order: List[str] = None,
    normalize_each: bool = True,
    height: int = 3
) -> None:
    height_chars = " ▁▂▃▄▅▆▇█"
    
    # Collect all keys if x_order not provided
    if x_order is None:
        all_keys = set()
        for dist in distributions:
            all_keys.update(dist.keys())
        x_order = sorted(all_keys)

    # Calculate spacing for labels
    MAX_LABEL_LEN = 4
    max_label_len = max(len(str(k)) for k in x_order) if x_order else 1
    spacing = min(MAX_LABEL_LEN, max_label_len)
    
    # Render rows
    rows = []

    # Calculate width needed for row numbers
    row_num_width = len(str(len(distributions)))

    for t, dist in enumerate(distributions):
        d = dist
        
        # Create multiple height levels for this distribution
        for h in range(height, 0, -1):  # Start from top
            text = Text()
            text.append(" "*row_num_width + " │ " if h != (height + 1) // 2 else f"{t:>{row_num_width}} │ ")
            
            for i, k in enumerate(x_order):
                v = d.get(k, 0)
                color = COLORS[i % len(COLORS)]
                # Calculate which character to show at this height level
                total_height = v * height
                if total_height >= h:
                    # Show full block at this level
                    bar_char = "█"
                elif total_height >= h - 1:
                    # Show partial block based on fractional part
                    fraction = total_height - (h - 1)
                    idx = min(int(fraction * (len(height_chars) - 1)), len(height_chars) - 1)
                    bar_char = height_chars[idx]
                else:
                    # Empty at this level
                    bar_char = " "
                
                # Center the bar character within the label space
                padding_before = (spacing - 1) // 2
                padding_after = spacing - 1 - padding_before
                text.append(bar_char * padding_before + bar_char + bar_char * padding_after, style=color)
            
            rows.append(text)
        
        # Add vertical spacing between distributions (except after the last one)
        if t < len(distributions) - 1:
            rows.append(Text(" "*row_num_width + " │" + '-'*row_num_width*spacing*(padding_before+padding_after)))

    # X-axis labels - full labels with proper spacing
    x_labels = Text(" "*row_num_width + " └ ")
    for i, k in enumerate(x_order):
        label = str(k)
        color = COLORS[i % len(COLORS)]
        
        # Truncate labels longer than 10 characters and add "..."
        if len(label) > MAX_LABEL_LEN:
            label = label[:MAX_LABEL_LEN-2] + ".."
        
        # Pad or truncate label to fit spacing
        if len(label) <= spacing:
            padded_label = label.ljust(spacing)
        else:
            padded_label = label[:spacing]
        x_labels.append(padded_label, style=color)
    
    rows.append(x_labels)
    for row in rows:
        console.print(row)


def to_discrete_distributions(
    values: List[List[str]],
    top_k: int = 10
) -> Tuple[List[Dict[str, float]], List[str]]:
    flat = [item for sublist in values for item in sublist]
    overall_counts = Counter(flat)
    top_categories = [k for k, _ in overall_counts.most_common(top_k)]
    all_categories = top_categories + ["other"]

    distributions = []
    for row in values:
        counts = Counter(k if k in top_categories else "other" for k in row)
        total = sum(counts.values()) or 1
        dist = {k: counts.get(k, 0) / total for k in all_categories}
        distributions.append(dist)

    return distributions, all_categories


if __name__ == '__main__':
    # Example with distributional data
    distributions = [
        {"cat": 0.2, "dog": 0.5, "fish": 0.3},
        {"cat": 0.4, "dog": 0.5, "fish": 0.2},
        {"cat": 0.1, "dog": 0.2, "fish": 0.7},
    ]

    x_order = ["cat", "dog", "fish"]
    ridgeline_plot(distributions, x_order)

    # Example with discrete data
    data = [
        ["cat", "dog", "dog", "fish", "lizard                   asd    "],
        ["cat", "snake", "dog", "fish"],
        ["hamster", "fish", "fish", "dog"],
    ]
    distributions, x_order = to_discrete_distributions(data)
    ridgeline_plot(distributions, x_order)
