#include <Python.h>

typedef struct Matrix {
  int rows;     // number of rows
  int cols;     // number of columns
  double *data; // pointer to rows * columns doubles
  int ref_cnt;  // How many slices/matrices are referring to this Matrix's data
  struct Matrix *parent; // NULL if Matrix is not a slice, else the parent
                         // Matrix of the slice
} Matrix;

double rand_double(double low, double high);
void rand_matrix(Matrix *result, unsigned int seed, double low, double high);
int allocate_matrix(Matrix **mat, int rows, int cols);
int allocate_matrix_ref(Matrix **mat, Matrix *from, int offset, int rows,
                        int cols);
void deallocate_matrix(Matrix *mat);
double get(Matrix *mat, int row, int col);
void set(Matrix *mat, int row, int col, double val);
void fill_matrix(Matrix *mat, double val);
int add_matrix(Matrix *result, Matrix *mat1, Matrix *mat2);
int sub_matrix(Matrix *result, Matrix *mat1, Matrix *mat2);
int mul_matrix(Matrix *result, Matrix *mat1, Matrix *mat2);
int pow_matrix(Matrix *result, Matrix *mat, int pow);
int neg_matrix(Matrix *result, Matrix *mat);
int abs_matrix(Matrix *result, Matrix *mat);
