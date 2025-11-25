# MatrixQR
A C#-implementation of simple vector and matrix operations. Currently support: 

- Vector addition, scalar multiplication, dot product, norm; 
- Matrix addition, scalar multiplication, matrix multiplication, transpose;
- Determinant;
- Inverse (Gaussian-Jordan elimination)
- LU decomposition *(when no row exchanges applied)*
- QR decomposition (Gram-Schmidt)

Efficiency of the algorithms are not considered (they are too complicated, I believe). There is much space for improvements. 

Several test matrices included; please refer to `Main` function. GitHub Copilot helped me generate these test cases. 

The file structure is a little messy. Don't mind. 

**This project is part of the final assignment of SJTU Linear Algebra (H) course.** *Author: Shengyuan Cai.* 
