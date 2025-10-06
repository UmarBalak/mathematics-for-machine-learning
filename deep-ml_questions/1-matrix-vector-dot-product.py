def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
    col_a = len(a[0])
    if col_a != len(b):
        return -1
    dot_prod_list = col_a * [0]
    for i in range(len(a[0])):
        for j in range(len(b)):
            dot_prod_list[i] += a[i][j] * b[j]
    return dot_prod_list