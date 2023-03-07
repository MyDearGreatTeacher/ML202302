A, B, C, D = np.meshgrid(range(1,4), range(1,4), range(1,4), range(1,4))
mesh = list(zip(A.flatten(), B.flatten(), C.flatten(), D.flatten()))
print(f'共{len(mesh)}個網格點，即3*3*3*3')