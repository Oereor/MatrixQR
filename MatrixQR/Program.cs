using System.Numerics;

namespace MatrixQR
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
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
                return GetDeterminant();
            }
        }
        
        private double GetDeterminant()
        {
            throw new NotImplementedException();
        }

        public bool IsInvertible
        {
            get { return IsSquare && Math.Abs(Determinant) > Tolerance; }
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

        public Matrix Invert()
        {
            throw new NotImplementedException();
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
            throw new NotImplementedException();
        }
    }
}
