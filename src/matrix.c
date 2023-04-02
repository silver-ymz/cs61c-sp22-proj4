#include "matrix.h"
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
 */

/* Generates a random double between low and high */
double rand_double(double low, double high) {
  double range = (high - low);
  double div = RAND_MAX / range;
  return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(Matrix *result, unsigned int seed, double low, double high) {
  srand(seed);
  for (int i = 0; i < result->rows; i++) {
    for (int j = 0; j < result->cols; j++) {
      set(result, i, j, rand_double(low, high));
    }
  }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in
 * row-major order.
 */
double get(Matrix *mat, int row, int col) {
  return mat->data[mat->cols * row + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(Matrix *mat, int row, int col, double val) {
  mat->data[mat->cols * row + col] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data
 * array and initialize all entries to be zeros. `parent` should be set to NULL
 * to indicate that this matrix is not a slice. You should also set `ref_cnt`
 * to 1. You should return -1 if either `rows` or `cols` or both have invalid
 * values. Return -2 if any call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(Matrix **mat, int rows, int cols) {
  // HINTS: Follow these steps.
  // 1. Check if the dimensions are valid. Return -1 if either dimension is not
  // positive.
  // 2. Allocate space for the new matrix struct. Return -2 if allocating memory
  // failed.
  // 3. Allocate space for the matrix data, initializing all entries to be 0.
  // Return -2 if allocating memory failed.
  // 4. Set the number of rows and columns in the matrix struct according to the
  // arguments provided.
  // 5. Set the `parent` field to NULL, since this matrix was not created from a
  // slice.
  // 6. Set the `ref_cnt` field to 1.
  // 7. Store the address of the allocated matrix struct at the location `mat`
  // is pointing at.
  // 8. Return 0 upon success.
  if (rows <= 0 || cols <= 0) {
    return -1;
  }

  Matrix *matrix = malloc(sizeof(Matrix));
  if (matrix == NULL) {
    return -2;
  }

  double *data = malloc(sizeof(double) * rows * cols);
  if (data == NULL) {
    free(mat);
    return -2;
  }
  for (int i = 0; i < rows * cols; i++) {
    data[i] = 0;
  }

  matrix->data = data;
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->ref_cnt = 1;
  matrix->parent = NULL;

  *mat = matrix;

  return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice
 * and has no existing slices, or that you free `mat->parent->data` if `mat` is
 * the last existing slice of its parent matrix and its parent matrix has no
 * other references (including itself).
 */
void deallocate_matrix(Matrix *mat) {
  // HINTS: Follow these steps.
  // 1. If the matrix pointer `mat` is NULL, return.
  // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the
  // `ref_cnt` field becomes 0, then free `mat` and its `data` field.
  // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then
  // free `mat`.
  if (mat == NULL) {
    return;
  }

  if (mat->parent != NULL) {
    deallocate_matrix(mat->parent);
    free(mat);
    return;
  }

  mat->ref_cnt--;
  if (mat->ref_cnt == 0) {
    free(mat->data);
    free(mat);
  }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and
 * `cols` columns. Its data should point to the `offset`th entry of `from`'s
 * data (you do not need to allocate memory) for the data field. `parent` should
 * be set to `from` to indicate this matrix is a slice of `from` and the
 * reference counter for `from` should be incremented. Lastly, do not forget to
 * set the matrix's row and column values as well. You should return -1 if
 * either `rows` or `cols` or both have invalid values. Return -2 if any call to
 * allocate memory in this function fails. Return 0 upon success. NOTE: Here
 * we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(Matrix **mat, Matrix *from, int offset, int rows,
                        int cols) {
  // Task 1.4 TODO
  // HINTS: Follow these steps.
  // 1. Check if the dimensions are valid. Return -1 if either dimension is not
  // positive.
  // 2. Allocate space for the new matrix struct. Return -2 if allocating memory
  // failed.
  // 3. Set the `data` field of the new struct to be the `data` field of the
  // `from` struct plus `offset`.
  // 4. Set the number of rows and columns in the new struct according to the
  // arguments provided.
  // 5. Set the `parent` field of the new struct to the `from` struct pointer.
  // 6. Increment the `ref_cnt` field of the `from` struct by 1.
  // 7. Store the address of the allocated matrix struct at the location `mat`
  // is pointing at.
  // 8. Return 0 upon success.
  if (rows <= 0 || cols <= 0) {
    return -1;
  }

  Matrix *matrix = malloc(sizeof(Matrix));
  if (matrix == NULL) {
    return -2;
  }

  matrix->data = from->data + offset;
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->parent = from;
  from->ref_cnt++;

  *mat = matrix;

  return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(Matrix *mat, double val) {
  if (mat == NULL) {
    return;
  }

  for (int i = 0; i < mat->rows * mat->cols; i++) {
    mat->data[i] = val;
  }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(Matrix *result, Matrix *mat) {
  if (mat == NULL || result == NULL || mat->rows != result->rows ||
      mat->cols != result->cols) {
    return -1;
  }

  for (int i = 0; i < mat->rows * mat->cols; i++) {
    result->data[i] = fabs(mat->data[i]);
  }

  return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(Matrix *result, Matrix *mat) {
  if (mat == NULL || result == NULL || mat->rows != result->rows ||
      mat->cols != result->cols) {
    return -1;
  }

  for (int i = 0; i < mat->rows * mat->cols; i++) {
    result->data[i] = -mat->data[i];
  }

  return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(Matrix *result, Matrix *mat1, Matrix *mat2) {
  if (mat1 == NULL || mat2 == NULL || result == NULL ||
      mat1->rows != mat2->rows || mat1->cols != mat2->cols ||
      mat1->rows != result->rows || mat1->cols != result->cols) {
    return -1;
  }

  for (int i = 0; i < mat1->rows * mat1->cols; i++) {
    result->data[i] = mat1->data[i] + mat2->data[i];
  }

  return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(Matrix *result, Matrix *mat1, Matrix *mat2) {
  if (mat1 == NULL || mat2 == NULL || result == NULL ||
      mat1->rows != mat2->rows || mat1->cols != mat2->cols ||
      mat1->rows != result->rows || mat1->cols != result->cols) {
    return -1;
  }

  for (int i = 0; i < mat1->rows * mat1->cols; i++) {
    result->data[i] = mat1->data[i] - mat2->data[i];
  }

  return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual
 * elements. You may assume `mat1`'s number of columns is equal to `mat2`'s
 * number of rows. Note that the matrix is in row-major order.
 */
int mul_matrix(Matrix *result, Matrix *mat1, Matrix *mat2) {
  if (mat1 == NULL || mat2 == NULL || result == NULL ||
      mat1->cols != mat2->rows || mat1->rows != result->rows ||
      mat2->cols != result->cols) {
    return -1;
  }

  for (int i = 0; i < mat1->rows; i++) {
    for (int j = 0; j < mat2->cols; j++) {
      double sum = 0;
      for (int k = 0; k < mat1->cols; k++) {
        sum += mat1->data[i * mat1->cols + k] * mat2->data[k * mat2->cols + j];
      }
      result->data[i * result->cols + j] = sum;
    }
  }

  return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise
 * multiplication. You may assume `mat` is a square matrix and `pow` is a
 * non-negative integer. Note that the matrix is in row-major order.
 */
int pow_matrix(Matrix *result, Matrix *mat, int pow) {
  if (mat == NULL || result == NULL || mat->rows != mat->cols ||
      mat->rows != result->rows || mat->cols != result->cols || pow < 0) {
    return -1;
  }

  Matrix *temp[2];
  allocate_matrix(&temp[0], mat->rows, mat->cols);
  allocate_matrix(&temp[1], mat->rows, mat->cols);

  bool flag = 0;

  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      temp[flag]->data[i * temp[flag]->cols + j] = mat->data[i * mat->cols + j];
    }
  }

  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      if (i == j) {
        result->data[i * result->cols + j] = 1;
      } else {
        result->data[i * result->cols + j] = 0;
      }
    }
  }

  for (int i = 0; i < 32; i++) {
    if ((pow & (1 << i)) != 0) {
      mul_matrix(temp[!flag], result, temp[flag]);
      for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
          result->data[i * result->cols + j] =
              temp[!flag]->data[i * temp[!flag]->cols + j];
        }
      }
    }

    if ((pow >> i) == 0) {
      break;
    }

    mul_matrix(temp[!flag], temp[flag], temp[flag]);
    flag = !flag;
  }

  deallocate_matrix(temp[0]);
  deallocate_matrix(temp[1]);

  return 0;
}
