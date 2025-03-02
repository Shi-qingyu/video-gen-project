import matplotlib.pyplot as plt

freq_dict = {
    'walk': 90, 'hold': 84, 'throw': 6, 'stand': 55, 'participate': 2,
    'drive': 51, 'compete': 2, 'transport': 1, 'race': 8, 'talk': 7,
    'interact': 6, 'mix': 1, 'wear': 27, 'work': 1, 'read': 1, 'grade': 1,
    'drink': 1, 'take': 8, 'give': 1, 'stretch': 1, 'play': 12, 'hit': 6,
    'bend': 6, 'adjust': 6, 'fix': 6, 'dance': 36, 'run': 13, 'engage': 1,
    'scale': 1, 'navigate': 15, 'secure': 1, 'descend': 1, 'perform': 20,
    'ride': 109, 'wind': 2, 'paraglide': 6, 'soar': 6, 'come': 3, 'rise': 3,
    'swirl': 2, 'trail': 2, 'flow': 1, 'skateboard': 5, 'push': 3, 'guide': 1,
    'support': 1, 'assist': 1, 'carry': 7, 'windsurf': 6, 'box': 1, 'observe': 6,
    'fence': 1, 'spar': 1, 'billow': 1, 'drift': 1, 'cycle': 1, 'negotiate': 1,
    'maneuver': 1, 'handle': 1, 'tackle': 1, 'rest': 1, 'shade': 1, 'travel': 6,
    'line': 1, 'paint': 1, 'adorn': 1
}

filtered_dict = {k: v for k, v in freq_dict.items() if v >= 5}
other_count = sum(v for v in freq_dict.values() if v < 5)
filtered_dict['other'] = other_count

labels = list(filtered_dict.keys())
sizes = list(filtered_dict.values())

fig, ax = plt.subplots(figsize=(10, 10))
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85)


for i, label in enumerate(labels):
    if label == 'other':
        autotexts[i].set_text('')

centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

ax.axis('equal')

plt.title("Motion Distribution of MTBench", fontsize=16)
plt.savefig("loop.jpg")
plt.show()