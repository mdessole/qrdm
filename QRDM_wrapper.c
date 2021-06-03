#include <python3.7m/Python.h>
#include <numpy/arrayobject.h>
#include <time.h>
#include <lapacke.h>
#include "QRDM.h"

double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);

PyObject* init(PyObject* self, PyObject* args)
{
  import_array();
  return Py_BuildValue("");
}

static PyObject* QP3(PyObject* self, PyObject* args)
{
  PyArrayObject *A_arr, *jpvt_arr, *tau_arr;
  double *A,  *tau;   // The C vectors to be created to point to the 
                            //   python vectors, cin and cout point to the row
                            //   of vecin and vecout, respectively
  int *jpvt;
  int matrix_layout,m,n,lda, out; 	

  /* Parse tuples separately since args will differ between C fcns */
  if (!PyArg_ParseTuple(args, "iiiOiOO", &matrix_layout, &m, &n,
			&A_arr, &lda, &jpvt_arr, &tau_arr))  return NULL;
  if (NULL == A_arr)  return NULL;
  if (NULL == jpvt_arr)  return NULL;
  if (NULL == tau_arr)  return NULL;
  A = pyvector_to_Carrayptrs(A_arr);
  jpvt = pyvector_to_Carrayptrs(jpvt_arr);
  tau = pyvector_to_Carrayptrs(tau_arr);
  

  //  clock_t start = clock();
  out = LAPACKE_dgeqp3(matrix_layout, m, n, A, lda, jpvt, tau);
  //clock_t stop = clock();
  //printf("C clock: Elapsed time  %.10f seconds\n", ((double)(stop-start))/CLOCKS_PER_SEC);
  
  return Py_BuildValue("i", out);
}



PyObject* QRDM(PyObject* self, PyObject* args)
{
  PyArrayObject *A_arr, *jpvt_arr, *tau_arr, *nc_arr, *thres_arr;
  double *A,  *tau, *thres;   // The C vectors to be created to point to the 
                            //   python vectors, cin and cout point to the row
                            //   of vecin and vecout, respectively
  int *jpvt, *nc;
  int matrix_layout,m,n,lda,out,DM_procedure; 	

  /* Parse tuples separately since args will differ between C fcns */
  if (!PyArg_ParseTuple(args, "iiiOiOOOOi", &matrix_layout, &m, &n,
			&A_arr, &lda, &jpvt_arr, &tau_arr,
			&nc_arr, &thres_arr, &DM_procedure))  return NULL;
  if (NULL == A_arr)  return NULL;
  if (NULL == jpvt_arr)  return NULL;
  if (NULL == tau_arr)  return NULL;
  if (NULL == nc_arr)  return NULL;
  if (NULL == thres_arr)  return NULL;
  A = pyvector_to_Carrayptrs(A_arr);
  jpvt = pyvector_to_Carrayptrs(jpvt_arr);
  tau = pyvector_to_Carrayptrs(tau_arr);
  nc = pyvector_to_Carrayptrs(nc_arr);
  thres = pyvector_to_Carrayptrs(thres_arr);
  //printf("Chiamo dgedmqr\n");
  //clock_t start = clock();
  out = dgeqrdm(matrix_layout, m, n, A, lda, jpvt, tau, nc, thres,DM_procedure);
  //clock_t stop = clock();
  //printf("C clock: Elapsed time  %.10f seconds\n", ((double)(stop-start))/CLOCKS_PER_SEC);

  return Py_BuildValue("i", out);
}


static PyObject* DORMQR(PyObject* self, PyObject* args)
{
  PyArrayObject *A_arr, *C_arr, *tau_arr;
  double *A, *C, *tau;   // The C vectors to be created to point to the 
                            //   python vectors, cin and cout point to the row
                            //   of vecin and vecout, respectively
  int matrix_layout,m,n,k,lda,ldc,out; 	

  /* Parse tuples separately since args will differ between C fcns */
  if (!PyArg_ParseTuple(args, "iiiiOiOOi", &matrix_layout,
			&m, &n, &k,
			&A_arr, &lda, &tau_arr, &C_arr, &ldc))  return NULL;
  if (NULL == A_arr)  return NULL;
  if (NULL == C_arr)  return NULL;
  if (NULL == tau_arr)  return NULL;
  A = pyvector_to_Carrayptrs(A_arr);
  C = pyvector_to_Carrayptrs(C_arr);
  tau = pyvector_to_Carrayptrs(tau_arr);
  
  out = LAPACKE_dormqr(matrix_layout, 'L', 'N', m, n, k,
		       A, lda, tau, C, ldc);
  return Py_BuildValue("i", out);
}



// Public interface
static PyMethodDef QRDMInterfaceMethods[] = {
 //{ "overwriteStateVariable",  overwriteStateVariable,  METH_VARARGS, "Overwrite a state variable"},
 { "init",  init,  METH_VARARGS, "initialize module"},
 { "QP3",  QP3,  METH_VARARGS, "LAPACK QR with column pivoting (from lapacke)"},
 { "QRDM",  QRDM,  METH_VARARGS, "QR with Deviation Maximization pivoting"},
 { "DORMQR", DORMQR, METH_VARARGS, "LAPACK dormqr" },
 { NULL, NULL, 0, NULL }
};


static struct PyModuleDef QRDMModule =
{
  PyModuleDef_HEAD_INIT,
  "QRDM", "With this wrapper you can run the QRDM algorithm from Python.",
  -1,
  QRDMInterfaceMethods
};

PyMODINIT_FUNC PyInit_QRDM(void)
{
  return PyModule_Create(&QRDMModule);
}

double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)
{
// 	int n=arrayin->dimensions[0];
	return (double *) arrayin->data;  /* pointer to arrayin data as double */
}
