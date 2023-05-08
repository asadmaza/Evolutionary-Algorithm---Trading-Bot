import random


def crossover_literals(dnf1, dnf2):
  # Make sure both DNF expressions have at least one conjunction
  if len(dnf1) == 0 or len(dnf2) == 0:
    raise ValueError(
        "Both DNF expressions must have at least one conjunction.")

  # Select a random conjunction from each DNF expression
  conj_index_dnf1 = random.randint(0, len(dnf1) - 1)
  conj_index_dnf2 = random.randint(0, len(dnf2) - 1)

  # Select a random literal from each chosen conjunction
  literal_index_dnf1 = random.randint(0, len(dnf1[conj_index_dnf1]) - 1)
  literal_index_dnf2 = random.randint(0, len(dnf2[conj_index_dnf2]) - 1)
  print(conj_index_dnf1)
  print(conj_index_dnf2)
  print(literal_index_dnf1)
  print(literal_index_dnf2)

  # Perform the crossover
  new_dnf1 = dnf1.copy()
  new_dnf2 = dnf2.copy()

  # Swap the literals between the two DNF expressions
  (
      new_dnf1[conj_index_dnf1][literal_index_dnf1],
      new_dnf2[conj_index_dnf2][literal_index_dnf2],
  ) = (
      new_dnf2[conj_index_dnf2][literal_index_dnf2],
      new_dnf1[conj_index_dnf1][literal_index_dnf1],
  )

  return new_dnf1, new_dnf2


dnf1 = [
    [["A", "B", "C"], ["D", "E", "F"]],
    [["G", "H", "I"]],
    [["J", "K", "L"], ["M", "N", "O"]],
]

dnf2 = [
    [["P", "Q", "R"]],
    [["S", "T", "U"], ["V", "W", "X"]],
    [["Y", "Z", "1"]],
]

while True:
  new_dnf1, new_dnf2 = crossover_literals(dnf1, dnf2)

  print("New DNF 1:", new_dnf1)
  print("New DNF 2:", new_dnf2)
