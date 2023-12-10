#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

persons = ['Farrah', 'Fred', 'Felicia']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

plt.figure()
bar_width = 0.5
bar_positions = np.arange(len(persons))

for i in range(fruit.shape[0]):
    plt.bar(persons, fruit[i], bottom=np.sum(fruit[:i], axis=0),
            color=colors[i], label=fruits[i], width=bar_width)

plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.ylim(0, 80)
plt.yticks(np.arange(0, 81, 10))
plt.xticks(bar_positions, persons)

plt.legend(["apples", "bananas", "oranges", "peaches"])

plt.show()
