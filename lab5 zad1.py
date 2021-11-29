#zad1
def znak(y):
    print('Wprowadź znak:')
y = ''
while type(y) != int:
    try:
        y = int(input())
        print('Wprowadzono liczbę całkowitą.')
    except ValueError:
        print('Wprowadzono błędny znak, wprowadź znak ponownie')