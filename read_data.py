import csv


def get_data(number: int) -> list:
    result = []
    with open(f'./dataset/data{number}.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) != 0:
                result.append((float(row[0]), float(row[1])))
    return result


def print_data(number: int) -> None:
    data = get_data(number)
    for row in data:
        print(row)
