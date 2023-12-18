import mpi.*;

public class MatrixMultiplication {

    public static void main(String[] args) throws MPIException {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        // Розмір матриць та їх ініціалізація
        int rowsA = 4;
        int colsA = 4;
        int rowsB = 4;
        int colsB = 4;
        int[][] matrixA = new int[rowsA][colsA];
        int[][] matrixB = new int[rowsB][colsB];
        int[][] result = new int[rowsA][colsB];

        // Ініціалізація матриць (призначте значення за необхідністю)
        initializeMatrix(matrixA);
        initializeMatrix(matrixB);

        // Розподіл роботи між процесами
        int[][] localMatrixA = distributeMatrix(matrixA, size, rank);
        int[][] localMatrixB = distributeMatrix(matrixB, size, rank);

        // Обчислення результату за вибраним алгоритмом
        switch (rank) {
            case 0:
                // Алгоритм1 - стрічкова схема
                result = sequentialAlgorithm(matrixA, matrixB);
                break;
            case 1:
                // Алгоритм2 - метод Фокса
                result = foxAlgorithm(localMatrixA, localMatrixB, size);
                break;
            case 2:
                // Алгоритм3 - метод Кеннона
                result = cannonAlgorithm(localMatrixA, localMatrixB, size);
                break;
            default:
                break;
        }

        // Збір результатів на головному процесі та виведення
        gatherAndPrintResult(result, size, rank);

        MPI.Finalize();
    }

    private static void initializeMatrix(int[][] matrix) {
        // Ініціалізувати матрицю значеннями за необхідності
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = i + j; // Приклад ініціалізації (замініть на власний код)
            }
        }
    }

    private static int[][] sequentialAlgorithm(int[][] matrixA, int[][] matrixB) {
        // Реалізація стрічкової схеми
        int[][] result = new int[matrixA.length][matrixB[0].length];
        for (int i = 0; i < matrixA.length; i++) {
            for (int j = 0; j < matrixB[0].length; j++) {
                for (int k = 0; k < matrixA[0].length; k++) {
                    result[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }
        return result;
    }

    private static int[][] distributeMatrix(int[][] matrix, int size, int rank) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        int rowsPerProcess = rows / size;
        int extraRows = rows % size;

        int[][] localMatrix;

        if (rank < extraRows) {
            // Процеси з залишковими рядками отримують на один рядок більше
            localMatrix = new int[rowsPerProcess + 1][cols];
        } else {
            localMatrix = new int[rowsPerProcess][cols];
        }

        int[] sendcounts = new int[size];
        int[] displacements = new int[size];

        for (int i = 0; i < size; i++) {
            sendcounts[i] = i < extraRows ? (rowsPerProcess + 1) * cols : rowsPerProcess * cols;
            displacements[i] = i * rowsPerProcess * cols + Math.min(i, extraRows) * cols;
        }

        MPI.COMM_WORLD.Scatterv(matrix, 0, sendcounts, displacements, MPI.INT, localMatrix, 0, localMatrix.length * cols, MPI.INT, 0);

        return localMatrix;
    }

    private static int[][] foxAlgorithm(int[][] matrixA, int[][] matrixB, int size) {
        int rank = MPI.COMM_WORLD.Rank();
        int rows = matrixA.length;
        int cols = matrixB[0].length;
        int blockSize = rows / size;

        int[][] localResult = new int[blockSize][cols];
        int[][] localMatrixA = new int[blockSize][cols];
        int[][] localMatrixB = new int[blockSize][cols];

        // Розподіл блоків матриць між процесами
        MPI.COMM_WORLD.Scatter(matrixA, 0, blockSize * cols, MPI.INT, localMatrixA, 0, blockSize * cols, MPI.INT, 0);
        MPI.COMM_WORLD.Scatter(matrixB, 0, blockSize * cols, MPI.INT, localMatrixB, 0, blockSize * cols, MPI.INT, 0);

        // Зміщення блоків матриць для обчислення
        int[][] shiftedMatrixA = new int[blockSize][cols];
        int[][] shiftedMatrixB = new int[blockSize][cols];

        // Копіюємо блоки в початкові позиції
        System.arraycopy(localMatrixA, 0, shiftedMatrixA, 0, blockSize);
        System.arraycopy(localMatrixB, 0, shiftedMatrixB, 0, blockSize);

        for (int stage = 0; stage < size; stage++) {
            // Обчислення локальної частини результату
            for (int i = 0; i < blockSize; i++) {
                for (int j = 0; j < blockSize; j++) {
                    for (int k = 0; k < cols; k++) {
                        localResult[i][j] += shiftedMatrixA[i][k] * shiftedMatrixB[j][k];
                    }
                }
            }

            // Зсув блоку матриці A вгору на один рядок
            int temp = shiftedMatrixA[0][0];
            for (int i = 0; i < blockSize - 1; i++) {
                System.arraycopy(shiftedMatrixA[i + 1], 0, shiftedMatrixA[i], 0, cols);
            }
            shiftedMatrixA[blockSize - 1][0] = temp;

            // Зсув блоку матриці B вліво на один стовпець
            temp = shiftedMatrixB[0][0];
            for (int i = 0; i < blockSize - 1; i++) {
                System.arraycopy(shiftedMatrixB[i], 1, shiftedMatrixB[i], 0, cols - 1);
            }
            shiftedMatrixB[0][blockSize - 1] = temp;

            // Обмін блоками між процесами
            MPI.COMM_WORLD.Sendrecv_replace(shiftedMatrixA, 0, blockSize * cols, MPI.INT, (rank + 1) % size, 0, (rank - 1 + size) % size, 0);
            MPI.COMM_WORLD.Sendrecv_replace(shiftedMatrixB, 0, blockSize * cols, MPI.INT, (rank + 1) % size, 1, (rank - 1 + size) % size, 1);
        }

        // Збір результатів на головному процесі
        int[][] result = new int[rows][cols];
        MPI.COMM_WORLD.Gather(localResult, 0, blockSize * cols, MPI.INT, result, 0, blockSize * cols, MPI.INT, 0);

        return result;
    }


    private static int[][] cannonAlgorithm(int[][] matrixA, int[][] matrixB, int size) {
        int rank = MPI.COMM_WORLD.Rank();
        int rows = matrixA.length;
        int cols = matrixB[0].length;
        int blockSize = rows / size;

        int[][] localResult = new int[blockSize][cols];
        int[][] localMatrixA = new int[blockSize][cols];
        int[][] localMatrixB = new int[blockSize][cols];

        // Розподіл блоків матриць між процесами
        MPI.COMM_WORLD.Scatter(matrixA, 0, blockSize * cols, MPI.INT, localMatrixA, 0, blockSize * cols, MPI.INT, 0);
        MPI.COMM_WORLD.Scatter(matrixB, 0, blockSize * cols, MPI.INT, localMatrixB, 0, blockSize * cols, MPI.INT, 0);

        // Виконання етапів обчислень
        for (int stage = 0; stage < size; stage++) {
            // Обчислення локальної частини результату
            for (int i = 0; i < blockSize; i++) {
                for (int j = 0; j < blockSize; j++) {
                    for (int k = 0; k < cols; k++) {
                        localResult[i][j] += localMatrixA[i][k] * localMatrixB[j][k];
                    }
                }
            }

            // Зсув блоку матриці A вліво на один стовпець
            int temp = localMatrixA[0][0];
            for (int i = 0; i < blockSize - 1; i++) {
                System.arraycopy(localMatrixA[i], 1, localMatrixA[i], 0, cols - 1);
            }
            localMatrixA[0][blockSize - 1] = temp;

            // Зсув блоку матриці B вгору на один рядок
            temp = localMatrixB[0][0];
            for (int i = 0; i < blockSize - 1; i++) {
                System.arraycopy(localMatrixB[i + 1], 0, localMatrixB[i], 0, cols);
            }
            localMatrixB[blockSize - 1][0] = temp;

            // Обмін блоками між процесами
            MPI.COMM_WORLD.Sendrecv_replace(localMatrixA, 0, blockSize * cols, MPI.INT, (rank + 1) % size, 0, (rank - 1 + size) % size, 0);
            MPI.COMM_WORLD.Sendrecv_replace(localMatrixB, 0, blockSize * cols, MPI.INT, (rank + 1) % size, 1, (rank - 1 + size) % size, 1);
        }

        // Збір результатів на головному процесі
        int[][] result = new int[rows][cols];
        MPI.COMM_WORLD.Gather(localResult, 0, blockSize * cols, MPI.INT, result, 0, blockSize * cols, MPI.INT, 0);

        return result;
    }


    private static void gatherAndPrintResult(int[][] result, int size, int rank) {
        int[][] gatheredResult = null;

        if (rank == 0) {
            // Головний процес отримує результати від інших процесів
            gatheredResult = new int[result.length][result[0].length];
        }

        // Ваш код для збору результатів на головному процесі
        MPI.COMM_WORLD.Gather(result, 0, result.length / size, MPI.INT, gatheredResult, 0, result.length / size, MPI.INT, 0);

        if (rank == 0) {
            // Виведення результату
            System.out.println("Final Result:");
            printMatrix(gatheredResult);
        }
    }

    private static void printMatrix(int[][] matrix) {
        // Вивести матрицю
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }
}