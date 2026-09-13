/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */


/*
* The t-SNE implementation was modified in with the modifications needed
* to become tsNET. The original t-SNE implementation had a bh-tree optimization
* method, which was removed for now.
*/

#ifndef TSNE_H
#define TSNE_H

#include "../Graph.h"
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cstring>
#include "tsNET.h"
#include <iostream>
#include "PivotMDS.h"
#include "../Utils.h"
#include "../Threading.h"

#include "tsNETTree/sptree.h"


namespace NNPNet {
	template<typename T>
    class TSNET
    {
    public:
        void tsNET(Graph<T>& g, double theta = 0, int iterations = 1500) {
            this->g = &g;
            if (theta == 0) {
                T* X = (T*)malloc(g.nodeCount * g.nodeCount * sizeof(T));

                for (int i = 0; i < g.nodeCount; i++) {
                    g.getDistances(i, X + (i * g.nodeCount));
					g.addFeaturesToDistances(i, featureWeight, X + (i * g.nodeCount));
                }

                double biggest = 0;
                for (int i = 0; i < g.nodeCount * g.nodeCount; i++) {
                    if (X[i] > biggest) {
                        biggest = X[i];
                    }
                }
                for (int i = 0; i < g.nodeCount * g.nodeCount; i++) {
                    X[i] /= biggest;
                }

                tsNET(X, g.Y, g.nodeCount, g.outputDim, iterations, false);
                free(X);
            }
            else {
                w_kl = 1;
                w_c = 1.2;
                w_r = 0;
                run(g.nodeCount, g.Y, g.outputDim, perp, theta, 5, false, iterations / 2, nullptr, 100);
            }
        };
        void tsNET(T* X, T* Y, int N, int D, int iterations, bool skipInit) {
            w_kl = 1;
            w_c = 1.2;
            w_r = 0;
            run(N, Y, D, perp, 0, 5, skipInit, iterations / 2, X, 100);
        };
        void tsNETStar(Graph<T>& g, double theta = 0, int iterations = 1500) {
            this->g = &g;
            PivotMDS<double> p;
            p.call(g);
            if (theta == 0) {
                T* X = (T*)malloc(g.nodeCount * g.nodeCount * sizeof(T));

				auto f = [&g, X, this](int s, int e) {
					for (int i = s; i < e; i++) {
						g.getDistances(i, X + (i * g.nodeCount));
						g.addFeaturesToDistances(i, featureWeight, X + (i * g.nodeCount));
					}
				};
				Threadpool::divideWork(f, g.nodeCount);

                double biggest = 0;
                for (int i = 0; i < g.nodeCount * g.nodeCount; i++) {
                    if (X[i] > biggest) {
                        biggest = X[i];
                    }
                }
                for (int i = 0; i < g.nodeCount * g.nodeCount; i++) {
                    X[i] /= biggest;
                }

                tsNETStar(X, g.Y, g.nodeCount, g.outputDim, iterations);
                free(X);
            }
            else {
                w_kl = 1;
                w_c = 0.1;
                w_r = 0;
                run(g.nodeCount, g.Y, g.outputDim, perp, theta, 5, true, iterations / 2, nullptr, -1);
            }
        };
        void tsNETStar(T* X, T* Y, int N, int D, int iterations) {
            w_kl = 1;
            w_c = 0.1;
            w_r = 0;
            run(N, Y, D, perp, 0, 5, true, iterations / 2, X, -1);
        };

		T perp = 40;
		T featureWeight = 0;

    private:
		inline T sign(T x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }
        void run(int N, T* Y, int no_dims, T perplexity, T theta, int rand_seed,
            bool skip_random_init, int max_iter = 1000, T* simalarity_matrix = nullptr,
            int stop_lying_iter = 250, int mom_switch_iter = 250) {
			// Set random seed
			if (skip_random_init != true) {
				if (rand_seed >= 0) {
					printf("Using random seed: %d\n", rand_seed);
					srand((unsigned int)rand_seed);
				}
				else {
					printf("Using current time as random seed...\n");
					srand(time(NULL));
				}
			}

			if (N - 1 < 3 * perplexity)
			{
				perplexity = (N - 1) / 3 - 1;
			}
			printf("Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);
			const bool exact = theta == 0;

			// Set learning parameters
			float total_time = .0;
			clock_t start, end;
			T momentum = .5, final_momentum = .7;
			T eta = 200.0;

			// Allocate some memory
			T* dY = (T*)malloc(N * no_dims * sizeof(T));
			T* uY = (T*)malloc(N * no_dims * sizeof(T));
			T* gains = (T*)malloc(N * no_dims * sizeof(T));

			if (exact) {
				this->DD = (T*)malloc(N * N * sizeof(T));
				this->Q = (T*)malloc(N * N * sizeof(T));
			}

			if (dY == NULL || uY == NULL || gains == NULL) { printf("Memory allocation failed!\n"); exit(1); }
			for (int i = 0; i < N * no_dims; i++)    uY[i] = .0;
			for (int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

			// Normalize input data (to prevent numerical problems)
			printf("Computing input similarities...\n");
			start = clock();

			// Compute input similarities for exact t-SNE
			T* P = nullptr; unsigned int* row_P = nullptr; unsigned int* col_P = nullptr; T* val_P = nullptr;
			if (exact) {

				// Compute similarities
				printf("Exact?");
				if (simalarity_matrix != nullptr) {
					P = simalarity_matrix;
				}
				else {
					P = (T*)malloc(N * N * sizeof(T));
					if (P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
				}
				computeGaussianPerplexity(N, P, perplexity);

				// Symmetrize input similarities
				printf("Symmetrizing...\n");
				int nN = 0;
				for (int n = 0; n < N; n++) {
					int mN = (n + 1) * N;
					for (int m = n + 1; m < N; m++) {
						P[nN + m] += P[mN + n];
						P[mN + n] = P[nN + m];
						mN += N;
					}
					nN += N;
				}

				T sum_P = .0;
				for (int i = 0; i < N * N; i++) sum_P += P[i];
				for (int i = 0; i < N * N; i++) P[i] /= sum_P;
			}
			else {

				// Compute asymmetric pairwise input similarities
				computeGaussianPerplexity(N, g->outputDim, &row_P, &col_P, &val_P, perplexity, (int)(3 * perplexity));

				// Symmetrize input similarities
				symmetrizeMatrix(&row_P, &col_P, &val_P, N);
				double sum_P = .0;
				for (int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
				for (int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;

				T* totD = (T*)malloc(N * sizeof(T));
				g->getDistances(0, totD);
				g->addFeaturesToDistances(0, featureWeight, totD);
				scaleFactor = 0;
				for (int i = 0; i < N; i++) {
					if (scaleFactor < totD[i]) {
						scaleFactor = totD[i];
					}
				}
				scaleFactor = 1 / scaleFactor;
				free(totD);
			}

			end = clock();
			// Lie about the P-values
			if (stop_lying_iter > 0) {
				if (exact) { for (int i = 0; i < N * N; i++)        P[i] *= 12.0; }
				else { for (int i = 0; i < row_P[N]; i++) val_P[i] *= 12.0; }
			}

			// Initialize solution (randomly)
			if (skip_random_init != true) {
				for (int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;
			}

			// Perform main training loop
			if (exact) printf("Input similarities computed in %4.2f seconds!\nLearning embedding...\n", (float)(end - start) / CLOCKS_PER_SEC);
			else printf("Input similarities computed in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float)(end - start) / CLOCKS_PER_SEC, (double)row_P[N] / ((double)N * (double)N));
			start = clock();

			T prevError = DBL_MAX;
			bool switched = false;
		phase3:
			for (int iter = 0; iter < max_iter; iter++) {

				// Compute (approximate) gradient
				if (exact) computeExactGradient(P, Y, N, no_dims, dY);
				else computeGradient(P, row_P, col_P, val_P, Y, N, no_dims, dY, theta);

				// Update gains
				for (int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
				for (int i = 0; i < N * no_dims; i++) if (gains[i] < .01) gains[i] = .01;

				// Perform gradient update (with momentum and gains)
				for (int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
				for (int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];

				// Make solution zero-mean
				zeroMean(Y, N, no_dims);

				// Stop lying about the P-values after a while, and switch momentum
				if (iter == stop_lying_iter) {
					if (exact) { for (int i = 0; i < N * N; i++)        P[i] /= 12.0; }
					else { for (int i = 0; i < row_P[N]; i++) val_P[i] /= 12.0; }
				}
				if (iter == mom_switch_iter) momentum = final_momentum;

				// Print out progress
				if ((iter % 50 == 0 || iter == max_iter - 1)) {
					end = clock();
					T C = .0;
					if (exact) C = evaluateError(P, Y, N, no_dims);
					else      C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta);  // doing approximate computation here!
					if (iter == 0)
						printf("Iteration %d: error is %f\n", iter + 1, C);
					else {
						total_time += (float)(end - start) / CLOCKS_PER_SEC;
						printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float)(end - start) / CLOCKS_PER_SEC);
					}
					if (prevError - C < 0.00001 && stop_lying_iter + 5 < iter) {
						if (switched) {
							std::cout << "Stopping early\n";
							break;
						}
						std::cout << "Switching...\n";
						w_kl = 1;
						w_c = 0.01;
						w_r = 0.6;
						iter = -1;
						switched = true;
						C = DBL_MAX;
						momentum = 0.5;
						stop_lying_iter = -1;
					}
					prevError = C;

					start = clock();
				}
			}
			if (!switched) {
				w_kl = 1;
				w_c = 0.01;
				w_r = 0.6;
				switched = true;
				momentum = 0.5;
				goto phase3;
			}
			end = clock(); total_time += (float)(end - start) / CLOCKS_PER_SEC;

			// Clean up memory
			free(dY);
			free(uY);
			free(gains);
			free(DD);
			free(Q);
			if (exact) {
				if (simalarity_matrix == nullptr) {
					free(P);
				}
			}
			else {
				free(row_P); row_P = NULL;
				free(col_P); col_P = NULL;
				free(val_P); val_P = NULL;
			}
			printf("Fitting performed in %4.2f seconds.\n", total_time);
		};
		void symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, T** _val_P, int N) {

			// Get sparse matrix
			unsigned int* row_P = *_row_P;
			unsigned int* col_P = *_col_P;
			T* val_P = *_val_P;

			// Count number of elements and row counts of symmetric matrix
			int* row_counts = (int*)calloc(N, sizeof(int));
			if (row_counts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
			for (int n = 0; n < N; n++) {
				for (int i = row_P[n]; i < row_P[n + 1]; i++) {

					// Check whether element (col_P[i], n) is present
					bool present = false;
					for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
						if (col_P[m] == n) present = true;
					}
					if (present) row_counts[n]++;
					else {
						row_counts[n]++;
						row_counts[col_P[i]]++;
					}
				}
			}
			int no_elem = 0;
			for (int n = 0; n < N; n++) no_elem += row_counts[n];

			// Allocate memory for symmetrized matrix
			unsigned int* sym_row_P = (unsigned int*)malloc((N + 1) * sizeof(unsigned int));
			unsigned int* sym_col_P = (unsigned int*)malloc(no_elem * sizeof(unsigned int));
			T* sym_val_P = (T*)malloc(no_elem * sizeof(T));
			if (sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

			// Construct new row indices for symmetric matrix
			sym_row_P[0] = 0;
			for (int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int)row_counts[n];

			// Fill the result matrix
			int* offset = (int*)calloc(N, sizeof(int));
			if (offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
			for (int n = 0; n < N; n++) {
				for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

					// Check whether element (col_P[i], n) is present
					bool present = false;
					for (unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
						if (col_P[m] == n) {
							present = true;
							if (n <= col_P[i]) {                                                 // make sure we do not add elements twice
								sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
								sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
								sym_val_P[sym_row_P[n] + offset[n]] = val_P[i] + val_P[m];
								sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
							}
						}
					}

					// If (col_P[i], n) is not present, there is no addition involved
					if (!present) {
						sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
						sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
						sym_val_P[sym_row_P[n] + offset[n]] = val_P[i];
						sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
					}

					// Update offsets
					if (!present || (present && n <= col_P[i])) {
						offset[n]++;
						if (col_P[i] != n) offset[col_P[i]]++;
					}
				}
			}

			// Divide the result by two
			for (int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

			// Return symmetrized matrices
			free(*_row_P); *_row_P = sym_row_P;
			free(*_col_P); *_col_P = sym_col_P;
			free(*_val_P); *_val_P = sym_val_P;

			// Free up some memery
			free(offset); offset = NULL;
			free(row_counts); row_counts = NULL;
		};

        void computeGradient(double* P, unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta) {
			// Construct space-partitioning tree on current map
			SPTree* tree = new SPTree(D, Y, N);
			tree->w_c = w_c;
			tree->w_kl = w_kl;
			tree->w_r = w_r;

			// Compute all terms required for t-SNE gradient
			double* pos_f = (double*)calloc(N * D, sizeof(double));
			double* neg_f = (double*)calloc(N * D, sizeof(double));
			if (pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
			tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
			
			auto f = [tree, theta, neg_f, D](int begin, int end) -> double {
				double sum_Q = 0;
				double* buff = (double*)malloc(sizeof(double) * tree->dimension);
				for (int n = begin; n < end; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q, buff);
				free(buff);
				return sum_Q;
			};
			double sum_Q = Threadpool::divideWork<double>(f, N, [](double q, double q2) {return q + q2; });


			// Compute final t-SNE gradient
			for (int i = 0; i < N * D; i++) {
				dC[i] = pos_f[i] - (neg_f[i] / sum_Q) + Y[i] * w_c / N;
			}
			free(pos_f);
			free(neg_f);
			delete tree;
		};

        void computeExactGradient(T* P, T* Y, int N, int D, T* dC) {

			// Make sure the current gradient contains zeros
			for (int i = 0; i < N * D; i++) dC[i] = 0.0;

			// Compute the squared Euclidean distance matrix
			if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
			computeSquaredEuclideanDistance(Y, N, D, DD);

			// Compute Q-matrix and normalization sum
			if (Q == NULL) { printf("Memory allocation failed!\n"); exit(2); }
			T sum_Q = Threadpool::divideWork<T>([this, N](int start, int end) {return computeGradiantPart1(N, start, end); }, N, [](T q0, T q1) {return q0 + q1; });

			// Perform the computation of the gradient
			Threadpool::divideWork([this, N, P, sum_Q, D, Y, dC](int start, int end) {computeGradiantPart2(N, P, sum_Q, D, Y, dC, start, end); }, N);

		}
        T computeGradiantPart1(int N, int start, int end) {
			T sum_Q = .0;
			for (int n = start; n < end; n++) {
				int nN = n * N;
				for (int m = 0; m < N; m++) {
					if (n != m) {
						Q[nN + m] = 1 / (1 + DD[nN + m]);
						sum_Q += Q[nN + m];
					}
				}
			}
			return sum_Q;
		};
        void computeGradiantPart2(int N, T* P, T sum_Q, int D, T* Y, T* dC, int start, int end) {
			int nD = 0;
			for (int n = start; n < end; n++) {
				int nN = n * N;
				int mD = 0;
				int nD = n * D;
				for (int m = 0; m < N; m++) {
					if (n != m) {
						T mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
						for (int d = 0; d < D; d++) {
							T dif = Y[nD + d] - Y[mD + d];
							dC[nD + d] += (dif)*mult * w_kl
								- 1 / (dif + ((dif > 0) ? 1.0 / 20 : -1.0 / 20)) * w_r / (2 * N * N);
						}
					}
					mD += D;
				}
				for (int d = 0; d < D; d++) {
					dC[nD + d] += Y[nD + d] * w_c / N;
				}
				nN += N;
				nD += D;
			}
		};

        T evaluateError(T* P, T* Y, int N, int D) {

			// Compute the squared Euclidean distance matrix
			T* DD = (T*)malloc(N * N * sizeof(T));
			T* Q = (T*)malloc(N * N * sizeof(T));
			if (DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
			computeSquaredEuclideanDistance(Y, N, D, DD);

			// Compute Q-matrix and normalization sum
			int nN = 0;
			T sum_Q = DBL_MIN;
			for (int n = 0; n < N; n++) {
				for (int m = 0; m < N; m++) {
					if (n != m) {
						Q[nN + m] = 1 / (1 + DD[nN + m]);
						sum_Q += Q[nN + m];
					}
					else Q[nN + m] = DBL_MIN;
				}
				nN += N;
			}
			for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

			// Sum t-SNE error
			T C = .0;
			for (int n = 0; n < N * N; n++) {
				C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
			}
			T compression = .0;
			int j = 0;
			for (int i = 0; i < N; i++) {
				T tot = 0;
				for (int d = 0; d < D; d++) {
					tot += Y[j] * Y[j];
					j++;
				}
				compression += tot;
			}
			compression /= (2 * N);

			T repulsion = .0;
			if (w_r != 0) {
				for (int n = 0; n < N * N; n++) {
					repulsion += log(DD[n] + 1.0 / 20.0);
				}
				repulsion /= (2 * N * N);
			}

			// Clean up memory
			free(DD);
			free(Q);
			return C * w_kl + compression * w_c + repulsion * w_r;
		};
        double evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta) {

			// Get estimate of normalization term
			SPTree* tree = new SPTree(D, Y, N);
			double* buff = (double*)calloc(D, sizeof(double));
			double* tB = (double*)malloc(sizeof(double) * tree->dimension);
			double sum_Q = .0;
			for (int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q, tB);
			free(tB);

			// Loop over all edges to compute t-SNE error
			int ind1, ind2;
			double C = .0, Q;

			T repulsion = .0;
			for (int n = 0; n < N; n++) {
				ind1 = n * D;
				for (int i = row_P[n]; i < row_P[n + 1]; i++) {
					Q = .0;
					ind2 = col_P[i] * D;
					for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
					for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
					for (int d = 0; d < D; d++) Q += buff[d] * buff[d];

					repulsion += log(sqrt(Q) + 1.0 / 20.0);

					Q = (1.0 / (1.0 + Q)) / sum_Q;
					C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
				}

			}
			repulsion /= (2 * N * N);
			T compression = .0;
			int j = 0;
			for (int i = 0; i < N; i++) {
				T tot = 0;
				for (int d = 0; d < D; d++) {
					tot += Y[j] * Y[j];
					j++;
				}
				compression += tot;
			}
			compression /= (2 * N);

			// Clean up memory
			free(buff);
			delete tree;
			return C * w_kl + compression * w_c + repulsion * w_r;
		};
        void zeroMean(T* X, int N, int D) {

			// Compute data mean
			T* mean = (T*)calloc(D, sizeof(T));
			if (mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
			int nD = 0;
			for (int n = 0; n < N; n++) {
				for (int d = 0; d < D; d++) {
					mean[d] += X[nD + d];
				}
				nD += D;
			}
			for (int d = 0; d < D; d++) {
				mean[d] /= (T)N;
			}

			// Subtract data mean
			nD = 0;
			for (int n = 0; n < N; n++) {
				for (int d = 0; d < D; d++) {
					X[nD + d] -= mean[d];
				}
				nD += D;
			}
			free(mean); mean = NULL;
		};
        void computeGaussianPerplexity(int N, T* P, T perplexity) {

			// Compute the squared Euclidean distance matrix
			T* DD = (T*)malloc(N * N * sizeof(T));
			if (DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
			std::memcpy(DD, P, N * N * sizeof(T));


			// Compute the Gaussian kernel row by row
			int nN = 0;
			for (int n = 0; n < N; n++) {

				// Initialize some variables
				bool found = false;
				T beta = 1.0;
				T min_beta = -DBL_MAX;
				T max_beta = DBL_MAX;
				T tol = 1e-5;
				T sum_P;

				// Iterate until we found a good perplexity
				int iter = 0;
				while (!found && iter < 200) {

					// Compute Gaussian kernel row
					for (int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
					P[nN + n] = DBL_MIN;

					// Compute entropy of current row
					sum_P = DBL_MIN;
					for (int m = 0; m < N; m++) sum_P += P[nN + m];
					T H = 0.0;
					for (int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
					H = (H / sum_P) + log(sum_P);

					// Evaluate whether the entropy is within the tolerance level
					T Hdiff = H - log(perplexity);
					if (Hdiff < tol && -Hdiff < tol) {
						found = true;
					}
					else {
						if (Hdiff > 0) {
							min_beta = beta;
							if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
								beta *= 2.0;
							else
								beta = (beta + max_beta) / 2.0;
						}
						else {
							max_beta = beta;
							if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
								beta /= 2.0;
							else
								beta = (beta + min_beta) / 2.0;
						}
					}

					// Update iteration counter
					iter++;
				}

				// Row normalize P
				for (int m = 0; m < N; m++) P[nN + m] /= sum_P;
				nN += N;
			}

			// Clean up memory
			free(DD); DD = NULL;
		};
        void computeGaussianPerplexity(int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K) {
			if (perplexity > K) printf("Perplexity should be lower than K!\n");

			// Allocate the memory we need
			*_row_P = (unsigned int*)malloc((N + 1) * sizeof(unsigned int));
			*_col_P = (unsigned int*)calloc(N * K, sizeof(unsigned int));
			*_val_P = (double*)calloc(N * K, sizeof(double));
			if (*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
			unsigned int* row_P = *_row_P;
			unsigned int* col_P = *_col_P;
			double* val_P = *_val_P;
			row_P[0] = 0;
			for (int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int)K;

			auto f = [row_P, col_P, val_P, N, this, K, perplexity](int begin, int end) {
				double* cur_P = (double*)malloc((N - 1) * sizeof(double));
				if (cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
				T* distances = (T*)malloc(K * sizeof(T));
				int* nodes = (int*)malloc(K * sizeof(int));
				for (int n = begin; n < end; n++) {

					if (n % 10000 == 0) printf(" - point %d of %d\n", n, N);

					// Find nearest neighbors
					if (featureWeight > 0 && g->featureDimensions > 0) {
						g->knnWithFeatures(n, nodes, distances, K, featureWeight);
					}
					else {
						g->knn(n, nodes, distances, K);
					}
					for (int i = 0; i < K; i++) {
						distances[i] *= scaleFactor;
					}

					// Initialize some variables for binary search
					bool found = false;
					double beta = 1.0;
					double min_beta = -DBL_MAX;
					double max_beta = DBL_MAX;
					double tol = 1e-5;

					// Iterate until we found a good perplexity
					int iter = 0; double sum_P;
					while (!found && iter < 200) {

						// Compute Gaussian kernel row
						for (int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m] * distances[m]);

						// Compute entropy of current row
						sum_P = DBL_MIN;
						for (int m = 0; m < K; m++) sum_P += cur_P[m];
						double H = .0;
						for (int m = 0; m < K; m++) H += beta * (distances[m] * distances[m] * cur_P[m]);
						H = (H / sum_P) + log(sum_P);

						// Evaluate whether the entropy is within the tolerance level
						double Hdiff = H - log(perplexity);
						if (Hdiff < tol && -Hdiff < tol) {
							found = true;
						}
						else {
							if (Hdiff > 0) {
								min_beta = beta;
								if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
									beta *= 2.0;
								else
									beta = (beta + max_beta) / 2.0;
							}
							else {
								max_beta = beta;
								if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
									beta /= 2.0;
								else
									beta = (beta + min_beta) / 2.0;
							}
						}

						// Update iteration counter
						iter++;
					}

					// Row-normalize current row of P and store in matrix
					for (unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
					for (unsigned int m = 0; m < K; m++) {
						col_P[row_P[n] + m] = nodes[m];
						val_P[row_P[n] + m] = cur_P[m];
					}
				}
				// Clean up memory
				free(cur_P);
				free(distances);
				free(nodes);
			};

			Threadpool::divideWork(f, N);



		};
        void computeSquaredEuclideanDistance(T* X, int N, int D, T* DD) {
			const T* XnD = X;
			for (int n = 0; n < N; ++n, XnD += D) {
				const T* XmD = XnD + D;
				T* curr_elem = &DD[n * N + n];
				*curr_elem = 0.0;
				T* curr_elem_sym = curr_elem + N;
				for (int m = n + 1; m < N; ++m, XmD += D, curr_elem_sym += N) {
					*(++curr_elem) = 0.0;
					for (int d = 0; d < D; ++d) {
						*curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
					}
					*curr_elem_sym = *curr_elem;
				}
			}
		};
        T randn() {
			T x, y, radius;
			do {
				x = 2 * (rand() / ((T)RAND_MAX + 1)) - 1;
				y = 2 * (rand() / ((T)RAND_MAX + 1)) - 1;
				radius = (x * x) + (y * y);
			} while ((radius >= 1.0) || (radius == 0.0));
			radius = sqrt(-2 * log(radius) / radius);
			x *= radius;
			y *= radius;
			return x;
		};

        T w_kl = 1, w_c = 1, w_r = 1;
        T* DD = nullptr, *Q = nullptr;

        Graph<T>* g = nullptr;
        T scaleFactor = 1;
    };
}

#endif
