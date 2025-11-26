using System.Numerics;
using System.Text.Json.Serialization;

namespace MatrixQR
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            // Run two tests: two 4x4 matrices (one invertible, one singular non-triangular)
            Console.WriteLine("Running Gaussian elimination inverse tests (two 4x4 matrices)...");

            double tol = 1e-8;

            void RunTestInverse(double[,] elems)
            {
                int n = elems.GetLength(0);
                Matrix m = new Matrix(elems);

                Console.WriteLine($"\nTest {n}x{n} matrix:");
                Console.WriteLine("Matrix m:\n" + m.ToString());

                try
                {
                    Matrix inv = m.Inverse();
                    Matrix prod = m * inv;
                    Matrix id = Matrix.Identity(n);

                    bool passed = true;
                    for (int i = 0; i < n; i++)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            if (Math.Abs(prod[i, j] - id[i, j]) > tol)
                            {
                                passed = false;
                                break;
                            }
                        }
                        if (!passed) break;
                    }

                    Console.WriteLine("Inverse inv:\n" + inv.ToString());
                    Console.WriteLine("Product m * inv:\n" + prod.ToString());
                    Console.WriteLine($"Inverse test passed: {passed}");
                }
                catch (InvalidOperationException)
                {
                    Console.WriteLine("Matrix is singular and cannot be inverted.");
                }
            }

            void RunTestQR(double[,] elems)
            {
                int n = elems.GetLength(0);
                Matrix m = new Matrix(elems);

                Console.WriteLine($"\nTest {n}x{n} matrix:");
                Console.WriteLine("Matrix m:\n" + m.ToString());

                try
                {
                    (Matrix Q, Matrix R) = Matrix.QRDecomposition(m);
                    Matrix prod = Q * R;

                    bool passed = true;
                    for (int i = 0; i < n; i++)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            if (Math.Abs(prod[i, j] - m[i, j]) > tol)
                            {
                                passed = false;
                                break;
                            }
                        }
                        if (!passed) break;
                    }

                    Console.WriteLine("Q matrix:\n" + Q.ToString());
                    Console.WriteLine("R matrix:\n" + R.ToString());
                    Console.WriteLine("Product Q * R:\n" + prod.ToString());
                    TestOrthogonality(Q);
                    Console.WriteLine($"Decomposition test passed: {passed}");
                }
                catch (InvalidOperationException)
                {
                    Console.WriteLine("Matrix cannot be QR decomposed.");
                }
            }

            void TestOrthogonality(Matrix Q)
            {
                int n = Q.Cols;
                bool orthogonal = true;
                for (int i = 0; i < n; i++)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        Vector colI = Q.GetColumn(i);
                        Vector colJ = Q.GetColumn(j);
                        if (!Vector.IsOrthogonal(colI, colJ))
                        {
                            orthogonal = false;
                            break;
                        }
                    }
                    if (!orthogonal) break;
                }
                Console.WriteLine($"Q matrix columns are orthogonal: {orthogonal}");
            }


            // QR decomposition tests: sizes 2, 3 and 4 with mostly-integer entries
            double[,] m2 = new double[,]
            {
                { 1, 2 },
                { 3, 4 }
            };

            double[,] m3 = new double[,]
            {
                { 1, 2, 3 },
                { 0, 1, 4 },
                { 5, 6, 0 }
            };

            double[,] m4 = new double[,]
            {
                { 0, 1, 0, 0 },
                { 0, 0, 1, 0 },
                { 0, 0, 0, 1 },
                { 1, 0, 0, 0 }
            };

            // Run inverse tests
            RunTestInverse(m2);
            RunTestInverse(m3);
            RunTestInverse(m4);

            // Run QR tests
            RunTestQR(m2);
            RunTestQR(m3);
            RunTestQR(m4);

            // LU decomposition tests (2x2, 3x3, 4x4) chosen so no row exchanges are required
            void RunTestLU(double[,] elems)
            {
                int n = elems.GetLength(0);
                Matrix m = new Matrix(elems);
                Console.WriteLine($"\nLU Test {n}x{n} matrix:");
                Console.WriteLine("Matrix m:\n" + m.ToString());
                try
                {
                    (Matrix L, Matrix U) = Matrix.LUDecomposition(new Matrix(elems));
                    Matrix prod = L * U;
                    bool passed = true;
                    for (int i = 0; i < n; i++)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            if (Math.Abs(prod[i, j] - m[i, j]) > tol)
                            {
                                passed = false;
                                break;
                            }
                        }
                        if (!passed) break;
                    }
                    Console.WriteLine("L matrix:\n" + L.ToString());
                    Console.WriteLine("U matrix:\n" + U.ToString());
                    Console.WriteLine("Product L * U:\n" + prod.ToString());
                    Console.WriteLine($"LU decomposition test passed: {passed}");
                }
                catch (InvalidOperationException ex)
                {
                    Console.WriteLine($"LU decomposition failed: {ex.Message}");
                }
            }

            void RunTestDeterminant(double[,] elems)
            {
                int n = elems.GetLength(0);
                Matrix m = new Matrix(elems);
                Console.WriteLine($"\nDeterminant Test {n}x{n} matrix:");
                Console.WriteLine("Matrix m:\n" + m.ToString());

                try
                {
                    Console.WriteLine("Determinant of matrix m: " + m.Determinant);
                }
                catch (InvalidOperationException ex)
                {
                    Console.WriteLine($"Failed to compute determinant: {ex.Message}");
                }
            }

            double[,] lu2 = new double[,]
            {
                { 1, 1 },
                { 2, 1 }
            };

            double[,] lu3 = new double[,]
            {
                { 2, 1, 1 },
                { 1, 3, 4 },
                { 1, 0, 4 }
            };

            double[,] lu4 = new double[,]
            {
                { 2, 1, 0, 0 },
                { 1, 3, 1, 0 },
                { 0, 2, 4, 1 },
                { 0, 0, 1, 5 }
            };

            RunTestLU(lu2);
            RunTestLU(lu3);
            RunTestLU(lu4);

            RunTestDeterminant(lu2);
            RunTestDeterminant(lu3);
            RunTestDeterminant(lu4);
        }
    }

    class Vector
    {
        private const double tolerance = 1e-10;

        private readonly int _size;

        private readonly double[] _elements;

        public Vector(int size)
        {
            if (size <= 0)
                throw new ArgumentException("Size must be positive.");
            _size = size;
            _elements = new double[size];
        }

        public Vector(double[] elements)
        {
            _size = elements.Length;
            _elements = new double[_size];
            Array.Copy(elements, _elements, _size);
        }

        public int Size
        {
            get { return _size; }
        }

        public double this[int index]
        {
            get { return _elements[index]; }
            set { _elements[index] = value; }
        }

        public static Vector operator +(Vector v1, Vector v2)
        {
            if (v1.Size != v2.Size)
                throw new ArgumentException("Vectors must be of the same size.");
            Vector result = new Vector(v1.Size);
            for (int i = 0; i < v1.Size; i++)
            {
                result[i] = v1[i] + v2[i];
            }
            return result;
        }

        public static Vector operator -(Vector self)
        {
            Vector result = new Vector(self.Size);
            for (int i = 0; i < self.Size; i++)
            {
                result[i] = -self[i];
            }
            return result;
        }

        public static Vector operator -(Vector v1, Vector v2)
        {
            if (v1.Size != v2.Size)
                throw new ArgumentException("Vectors must be of the same size.");
            Vector result = new Vector(v1.Size);
            for (int i = 0; i < v1.Size; i++)
            {
                result[i] = v1[i] - v2[i];
            }
            return result;
        }

        public static Vector operator *(double scalar, Vector v)
        {
            Vector result = new Vector(v.Size);
            for (int i = 0; i < v.Size; i++)
            {
                result[i] = scalar * v[i];
            }
            return result;
        }

        public static Vector operator *(Vector v, double scalar)
        {
            return scalar * v;
        }

        public static double operator *(Vector v1, Vector v2)
        {
            return DotProduct(v1, v2);
        }

        public static double DotProduct(Vector v1, Vector v2)
        {
            if (v1.Size != v2.Size)
                throw new ArgumentException("Vectors must be of the same size.");
            double result = 0;
            for (int i = 0; i < v1.Size; i++)
            {
                result += v1[i] * v2[i];
            }
            return result;
        }

        public double Norm
        {
            get { return Math.Sqrt(DotProduct(this, this)); }
        }

        public static Vector Zero(int size)
        {
            return new Vector(size);
        }

        public override string ToString()
        {
            return "[" + string.Join(", ", _elements) + "]";
        }

        public double[] ToArray()
        {
            double[] array = new double[_size];
            Array.Copy(_elements, array, _size);
            return array;
        }

        public Vector Normalize()
        {
            double norm = Norm;
            if (norm == 0)
                throw new InvalidOperationException("Cannot normalize a zero vector.");
            return (1.0 / norm) * this;
        }

        public static bool IsOrthogonal(Vector v1, Vector v2)
        {
            return Math.Abs(DotProduct(v1, v2)) < tolerance;
        }
    }

    class Matrix
    {
        private const double Tolerance = 1e-10;

        private readonly double[,] _elements;

        private readonly int _rows;
        private readonly int _cols;

        public Matrix(int rows, int cols)
        {
            if (rows <= 0 || cols <= 0)
                throw new ArgumentException("Number of rows and columns must be positive.");
            _rows = rows;
            _cols = cols;
            _elements = new double[rows, cols];
        }

        public Matrix(double[,] elements)
        {
            _rows = elements.GetLength(0);
            _cols = elements.GetLength(1);
            _elements = new double[_rows, _cols];
            Array.Copy(elements, _elements, elements.Length);
        }

        public int Rows
        {
            get { return _rows; }
        }

        public int Cols
        {
            get { return _cols; }
        }

        public double this[int row, int col]
        {
            get { return _elements[row, col]; }
            set { _elements[row, col] = value; }
        }

        public bool IsSquare
        {
            get { return _rows == _cols; }
        }

        public double Determinant
        {
            get
            {
                if (!IsSquare)
                {
                    throw new InvalidOperationException("Determinant is only defined for square matrices.");
                }
                return GetDeterminant();
            }
        }

        private double GetDeterminant()
        {
            Matrix copy = new Matrix(_elements);
            copy.GaussianElimination(out bool _, out int exchanges, out double[] diagonals);
            double det = 1.0;
            for (int i = 0; i < _rows; i++)
            {
                det *= diagonals[i];
            }
            return det * (exchanges % 2 == 0 ? 1 : -1);
        }

        public bool IsInvertible
        {
            get
            {
                return IsSquare && GetDeterminant() > Tolerance;
            }
        }

        public Vector GetRow(int row)
        {
            double[] elements = new double[_cols];
            for (int j = 0; j < _cols; j++)
            {
                elements[j] = _elements[row, j];
            }
            return new Vector(elements);
        }

        public void SetRow(int row, Vector vector)
        {
            if (vector.Size != _cols)
                throw new ArgumentException("Vector size must match the number of columns.");
            for (int j = 0; j < _cols; j++)
            {
                _elements[row, j] = vector[j];
            }
        }

        public Vector GetColumn(int col)
        {
            double[] elements = new double[_rows];
            for (int i = 0; i < _rows; i++)
            {
                elements[i] = _elements[i, col];
            }
            return new Vector(elements);
        }

        public void SetColumn(int col, Vector vector)
        {
            if (vector.Size != _rows)
                throw new ArgumentException("Vector size must match the number of rows.");
            for (int i = 0; i < _rows; i++)
            {
                _elements[i, col] = vector[i];
            }
        }

        public double[,] ToArray()
        {
            double[,] array = new double[_rows, _cols];
            Array.Copy(_elements, array, _elements.Length);
            return array;
        }

        public static Matrix Identity(int size)
        {
            Matrix identity = new Matrix(size, size);
            for (int i = 0; i < size; i++)
            {
                identity[i, i] = 1;
            }
            return identity;
        }

        public static Matrix Zero(int rows, int cols)
        {
            return new Matrix(rows, cols);
        }

        public static Matrix operator +(Matrix m1, Matrix m2)
        {
            if (m1.Rows != m2.Rows || m1.Cols != m2.Cols)
                throw new ArgumentException("Matrices must have the same dimensions.");
            Matrix result = new Matrix(m1.Rows, m1.Cols);
            for (int i = 0; i < m1.Rows; i++)
            {
                for (int j = 0; j < m1.Cols; j++)
                {
                    result[i, j] = m1[i, j] + m2[i, j];
                }
            }
            return result;
        }

        public static Matrix operator -(Matrix self)
        {
            Matrix result = new Matrix(self.Rows, self.Cols);
            for (int i = 0; i < self.Rows; i++)
            {
                for (int j = 0; j < self.Cols; j++)
                {
                    result[i, j] = -self[i, j];
                }
            }
            return result;
        }

        public static Matrix operator -(Matrix m1, Matrix m2)
        {
            if (m1.Rows != m2.Rows || m1.Cols != m2.Cols)
                throw new ArgumentException("Matrices must have the same dimensions.");
            Matrix result = new Matrix(m1.Rows, m1.Cols);
            for (int i = 0; i < m1.Rows; i++)
            {
                for (int j = 0; j < m1.Cols; j++)
                {
                    result[i, j] = m1[i, j] - m2[i, j];
                }
            }
            return result;
        }

        public static Matrix operator *(double scalar, Matrix m)
        {
            Matrix result = new Matrix(m.Rows, m.Cols);
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Cols; j++)
                {
                    result[i, j] = scalar * m[i, j];
                }
            }
            return result;
        }

        public static Matrix operator *(Matrix m, double scalar)
        {
            return scalar * m;
        }

        public static Matrix operator *(Matrix m1, Matrix m2)
        {
            if (m1.Cols != m2.Rows)
                throw new ArgumentException("Number of columns of the first matrix must equal the number of rows of the second matrix.");
            Matrix result = new Matrix(m1.Rows, m2.Cols);
            for (int i = 0; i < m1.Rows; i++)
            {
                for (int j = 0; j < m2.Cols; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < m1.Cols; k++)
                    {
                        sum += m1[i, k] * m2[k, j];
                    }
                    result[i, j] = sum;
                }
            }
            return result;
        }

        public Matrix Transpose()
        {
            Matrix transposed = new Matrix(_cols, _rows);
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    transposed[j, i] = _elements[i, j];
                }
            }
            return transposed;
        }

        public Vector[] ToRowVectorArray()
        {
            Vector[] rows = new Vector[_rows];
            for (int i = 0; i < _rows; i++)
            {
                rows[i] = GetRow(i);
            }
            return rows;
        }

        public Vector[] ToColumnVectorArray()
        {
            Vector[] cols = new Vector[_cols];
            for (int j = 0; j < _cols; j++)
            {
                cols[j] = GetColumn(j);
            }
            return cols;
        }

        public static (Matrix Q, Matrix R) QRDecomposition(Matrix m)
        {
            int rows = m.Rows;
            int cols = m.Cols;
            Matrix Q = new Matrix(rows, cols);
            Matrix R = new Matrix(rows, rows);

            for (int i = 0; i < cols; i++)
            {
                Vector a_i = m.GetColumn(i);
                Vector u_i = m.GetColumn(i);
                for (int j = 0; j < i; j++)
                {
                    Vector u_j = Q.GetColumn(j);
                    u_i -= Vector.DotProduct(a_i, u_j) / Vector.DotProduct(u_j, u_j) * u_j;
                }
                Q.SetColumn(i, u_i.Normalize());
                R[i, i] = u_i.Norm;
            }

            for (int i = 0; i < rows; i++)
            {
                for (int j = i + 1; j < cols; j++)
                {
                    R[i, j] = Vector.DotProduct(Q.GetColumn(i), m.GetColumn(j));
                }
            }

            return (Q, R);
        }

        /// <summary>
        /// Performs Gaussian elimination on the matrix.
        /// </summary>
        /// <returns>A bool value indicating whether the matrix has inverse. </returns>
        private void GaussianElimination(out bool canInvert, out int exchanges, out double[] diagonals)
        {
            double[] d = new double[_rows];
            bool hasInverse = true;
            int swaps = 0;
            for (int i = 0; i < _rows; i++)
            {
                for (int j = i; j < _rows; j++)
                {
                    if (Math.Abs(_elements[j, i]) > Math.Abs(_elements[i, i]))
                    {
                        ExchangeRows(i, j);
                        ++swaps;
                    }
                }
                if (Math.Abs(_elements[i, i]) < Tolerance)
                {
                    hasInverse = false;
                    continue;
                }
                d[i] = _elements[i, i];
                NormalizeRow(i, i);
                for (int j = i + 1; j < _rows; j++)
                {
                    SubtractRows(j, i, _elements[j, i]);
                }
            }
            canInvert = hasInverse;
            exchanges = swaps;
            diagonals = d;
        }

        private void BackSubstitution()
        {
            for (int i = _rows - 1; i >= 0; i--)
            {
                for (int j = i - 1; j >= 0; j--)
                {
                    SubtractRows(j, i, _elements[j, i]);
                }
            }
        }

        private void ExchangeRows(int row1, int row2)
        {
            for (int j = 0; j < _cols; j++)
            {
                double temp = _elements[row1, j];
                _elements[row1, j] = _elements[row2, j];
                _elements[row2, j] = temp;
            }
        }

        private void SubtractRows(int targetRow, int sourceRow, double factor)
        {
            for (int j = 0; j < _cols; j++)
            {
                _elements[targetRow, j] -= factor * _elements[sourceRow, j];
            }
        }

        public Matrix Inverse()
        {
            if (!IsSquare)
            {
                throw new InvalidOperationException("Only square matrices can be inverted.");
            }

            Matrix copy = new Matrix(_elements);

            Matrix augmented = new Matrix(_rows, 2 * _cols);
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    augmented[i, j] = copy[i, j];
                }
                augmented[i, i + _cols] = 1.0;
            }

            augmented.GaussianElimination(out bool canInvert, out int _, out double[] _);
            if (!canInvert)
            {
                throw new InvalidOperationException("Matrix is singular and cannot be inverted.");
            }
            augmented.BackSubstitution();

            Matrix res = new Matrix(_rows, _cols);
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    res[i, j] = augmented[i, j + _cols];
                }
            }
            return res;
        }

        private void NormalizeRow(int row, int pivotCol)
        {
            if (Math.Abs(_elements[row, pivotCol]) < Tolerance)
            {
                return;
            }

            double factor = _elements[row, pivotCol];
            for (int i = 0; i < _cols; i++)
            {
                _elements[row, i] /= factor;
            }
        }

        public override string ToString()
        {
            ClearFloatError();
            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            for (int i = 0; i < _rows; i++)
            {
                sb.Append("[");
                for (int j = 0; j < _cols; j++)
                {
                    sb.Append(_elements[i, j].ToString("G"));
                    if (j < _cols - 1) sb.Append(", ");
                }
                sb.Append("]");
                if (i < _rows - 1) sb.AppendLine();
            }
            return sb.ToString();
        }

        private void ClearFloatError()
        {
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    double element = _elements[i, j];
                    double round = Math.Round(element);
                    if (Math.Abs(element - round) < Tolerance)
                    {
                        _elements[i, j] = round;
                    }
                }
            }
        }

        private static Matrix PermutationSquare(int size, int row1, int row2)
        {
            if (row1 >= size || row2 >= size || row1 < 0 || row2 < 0)
                throw new ArgumentException("Row indices must be within matrix size.");

            Matrix perm = Identity(size);
            perm.ExchangeRows(row1, row2);
            return perm;
        }

        private static Matrix EliminationSquare(int size, int targetRow, int sourceRow, double factor)
        {
            if (targetRow >= size || sourceRow >= size || targetRow < 0 || sourceRow < 0)
                throw new ArgumentException("Row indices must be within matrix size.");

            Matrix elim = Identity(size);
            elim[targetRow, sourceRow] = -factor;
            return elim;
        }

        private static Matrix ScaleSquare(int size, int row, double factor)
        {
            if (row >= size || row < 0)
                throw new ArgumentException("Row index must be within matrix size.");
            Matrix scale = Identity(size);
            scale[row, row] = factor;
            return scale;
        }

        /// <summary>
        /// Performs Gaussian elimination on a square matrix and returns the transformation matrix.
        /// </summary>
        /// <returns>A matrix representing all elimination processes. </returns>
        /// <exception cref="InvalidOperationException"></exception>
        private Matrix GaussianEliminationOnSquare(out bool hasRowExchanges)
        {
            bool rowExchanged = false;
            if (!IsSquare)
            {
                throw new InvalidOperationException("Matrix must be square for this operation.");
            }

            Matrix res = Identity(_rows);
            for (int i = 0; i < _rows; i++)
            {
                if (Math.Abs(_elements[i, i]) < Tolerance)
                {
                    rowExchanged = true;
                }
                for (int j = i; j < _rows; j++)
                {
                    if (Math.Abs(_elements[j, i]) > Math.Abs(_elements[i, i]))
                    {
                        ExchangeRows(i, j);
                        res = PermutationSquare(_rows, i, j) * res;
                    }
                }
                if (Math.Abs(_elements[i, i]) < Tolerance)
                {
                    rowExchanged = false;
                    continue;
                }
                res = ScaleSquare(_rows, i, 1.0 / _elements[i, i]) * res;
                NormalizeRow(i, i);
                for (int j = i + 1; j < _rows; j++)
                {
                    res = EliminationSquare(_rows, j, i, _elements[j, i]) * res;
                    SubtractRows(j, i, _elements[j, i]);
                }
            }
            hasRowExchanges = rowExchanged;
            return res;
        }

        private Matrix BackSubstitutionOnSquare()
        {
            if (!IsSquare)
            {
                throw new InvalidOperationException("Matrix must be square for this operation.");
            }

            Matrix res = Identity(_rows);
            for (int i = _rows - 1; i >= 0; i--)
            {
                for (int j = i - 1; j >= 0; j--)
                {
                    res = EliminationSquare(_rows, j, i, _elements[j, i]) * res;
                    SubtractRows(j, i, _elements[j, i]);
                }
            }
            return res;
        }

        public static (Matrix L, Matrix U) LUDecomposition(Matrix m)
        {
            if (!m.IsSquare)
            {
                throw new InvalidOperationException("Matrix must be square for LU decomposition.");
            }

            Matrix copy = new Matrix(m._elements);
            Matrix L = copy.GaussianEliminationOnSquare(out bool hasRowExchanges);
            if (!hasRowExchanges)
            {
                Matrix U = copy.BackSubstitutionOnSquare();
                return (L.Inverse(), U.Inverse());
            }
            else
            {
                throw new InvalidOperationException("LU decomposition failed due to row exchanges.");
            }
        }
    }
}
