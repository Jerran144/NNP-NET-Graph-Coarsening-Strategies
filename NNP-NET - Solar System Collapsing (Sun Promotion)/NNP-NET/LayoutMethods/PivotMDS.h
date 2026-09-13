/** \file
 * \brief Declaration of the pivot MDS. By setting the number of pivots to
 * infinity this algorithm behaves just like classical MDS.
 * See Brandes and Pich: Eigensolver methods for progressive multidi-
 * mensional scaling of large data.
 *
 * \author Mark Ortmann, University of Konstanz
 *
 * \par License:
 * This file is part of the Open Graph Drawing Framework (OGDF).
 *
 * \par
 * Copyright (C)<br>
 * See README.md in the OGDF root directory for details.
 *
 * \par
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * Version 2 or 3 as published by the Free Software Foundation;
 * see the file LICENSE.txt included in the packaging of this file
 * for details.
 *
 * \par
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * \par
 * You should have received a copy of the GNU General Public
 * License along with this program; if not, see
 * http://www.gnu.org/copyleft/gpl.html
 */

 /*
 * The version of PivotMDS used in this project is a modified version
 * of the implementation from Open Graph Drawing Framework (OGDF).
 * Of note is the support for more than 3 output dimensions.
 */

#pragma once

#include <vector>
#include <limits>
#include "../Graph.h"
#include "../Utils.h"
#include "../Threading.h"

#include <cfloat>
#include <cmath>
#include <math.h>
#include <cstring>

#define PROCRUSTES
#include "../Procrustes.h"

namespace NNPNet {

	template<typename _T>
	inline bool isinf(_T value) {
		return std::numeric_limits<_T>::has_infinity && value == std::numeric_limits<_T>::infinity();
	}

	//! The Pivot MDS (multi-dimensional scaling) layout algorithm.
	/**
	 * @ingroup gd-energy
	 */
	template<typename T>
	class PivotMDS {
	public:


		PivotMDS()
			: m_numberOfPivots(250)
			, m_edgeCosts(100)
			, m_hasEdgeCostsAttribute(false)
			, m_forcing2DLayout(false) {
		}

		~PivotMDS() {}

		//! Sets the number of pivots. If the new value is smaller or equal 0
		//! the default value (250) is used.
		void setNumberOfPivots(int numberOfPivots) {
			m_numberOfPivots = numberOfPivots;
		}

		//! Sets the desired distance between adjacent nodes. If the new value is smaller or equal
		//! 0 the default value (100) is used.
		void setEdgeCosts(double edgeCosts) { m_edgeCosts = edgeCosts; }

		//! Sets whether a 2D-layout should be calculated even when
		//! GraphAttributes::threeD is set.
		void setForcing2DLayout(bool forcing2DLayout) { m_forcing2DLayout = forcing2DLayout; }

		//! Returns whether a 2D-layout is calculated even when
		//! GraphAttributes::threeD is set.
		bool isForcing2DLayout() const { return m_forcing2DLayout; }

		//! Calls the layout algorithm for graph attributes \p GA.
		/**
		 * Calculates a 3D-layout if GraphAttributes::threeD is set for \p GA, and a
		 * 2D-layout otherwise.
		 * You can use setForcing2DLayout() to force the calculation of a 2D-layout
		 * even when GraphAttributes::threeD is set.
		 */
		void call(Graph<T>& GA) {
			pivotMDSLayout(GA);
		};

		int* call(Graph<T>& GA, bool* inAll) {
			return pivotMDSLayout(GA, inAll);
		};

		void call(Graph<T>& GA, int* pivotPoints) {
			pivotMDSLayout(GA, nullptr, pivotPoints);
		};

		void useEdgeCostsAttribute(bool useEdgeCostsAttribute) {
			m_hasEdgeCostsAttribute = useEdgeCostsAttribute;
		}

		bool useEdgeCostsAttribute() const { return m_hasEdgeCostsAttribute; }

	private:
		//! Convergence factor used for power iteration.
		double EPSILON = 1 - 1e-10;

		//! Factor used to center the pivot matrix.
		double FACTOR = -0.5;

		//! Seed of the random number generator.
		const static unsigned int SEED = 0;

		//! The number of pivots.
		int m_numberOfPivots;

		//! The number of dimensions
		int m_dimensionCount;

		//! The costs to traverse an edge.
		T m_edgeCosts;

		// For time series data
#ifdef PROCRUSTES
		double* m_prevEvecs = nullptr;
#else
		std::vector<std::vector<double>> m_prevEvecs = {};
#endif

		//! Tells whether the pivot mds is based on uniform edge costs or a
		//! edge costs attribute
		bool m_hasEdgeCostsAttribute;

		//! Whether a 2D-layout is calculated even when
		//! GraphAttributes::threeD is set.
		bool m_forcing2DLayout;

		//! Centers the pivot matrix.
		void centerPivotmatrix(std::vector<std::vector<T>>& pivotMatrix) {
			int numberOfPivots = pivotMatrix.size();
			// this is ensured since the graph size is at least 2!
			int nodeCount = pivotMatrix[0].size();

			T normalizationFactor = 0;
			T rowColNormalizer;
			std::vector<T> colNormalization(numberOfPivots);

			for (int i = 0; i < numberOfPivots; i++) {
				rowColNormalizer = 0;
				for (int j = 0; j < nodeCount; j++) {
					rowColNormalizer += pivotMatrix[i][j] * pivotMatrix[i][j];
				}
				normalizationFactor += rowColNormalizer;
				colNormalization[i] = rowColNormalizer / nodeCount;
			}
			normalizationFactor = normalizationFactor / (nodeCount * numberOfPivots);
			for (int i = 0; i < nodeCount; i++) {
				rowColNormalizer = 0;
				for (int j = 0; j < numberOfPivots; j++) {
					T square = pivotMatrix[j][i] * pivotMatrix[j][i];
					pivotMatrix[j][i] = square + normalizationFactor - colNormalization[j];
					rowColNormalizer += square;
				}
				rowColNormalizer /= numberOfPivots;
				for (int j = 0; j < numberOfPivots; j++) {
					pivotMatrix[j][i] = FACTOR * (pivotMatrix[j][i] - rowColNormalizer);
				}
			}
		};

		//! Computes the pivot mds layout of the given connected graph of \p GA.
		int* pivotMDSLayout(Graph<T>& G, bool* inAll = nullptr, int* pivotPoints = nullptr) {
			//m_dimensionCount = GA.has(Graph::threeD) && !m_forcing2DLayout ? 3 : 2;
			bool use3D = false;
			m_dimensionCount = G.outputDim;

			const int n = G.nodeCount;
			std::vector<std::vector<double>> coord(m_dimensionCount);

			// trivial cases
			if (n == 0) {
				return nullptr;
			}

			if (n == 1) {
				for (int i = 0; i < n; i++) {
					G.Y[i] = 0;
				}
				return nullptr;
			}

			std::vector<std::vector<T>> pivDistMatrix;
			// compute the pivot matrix
			int* choosenPivots = nullptr;
			if (pivotPoints != nullptr) {
				getPivotDistanceMatrix(G, pivotPoints, pivDistMatrix);
			}
			else {
				choosenPivots = getPivotDistanceMatrix(G, pivDistMatrix, inAll);
			}
			// center the pivot matrix
			centerPivotmatrix(pivDistMatrix);
			// init the coordinate matrix
			for (auto& elem : coord) {
				elem.resize(n);
			}
			// init the eigen values std::vector
			std::vector<double> eVals(m_dimensionCount);
			singularValueDecomposition(pivDistMatrix, coord, eVals, pivotPoints != nullptr || inAll != nullptr);
			// compute the correct aspect ratio
			for (int i = 0; i < coord.size(); i++) {
				eVals[i] = sqrt(eVals[i]);
				for (int j = 0; j < n; j++) {
					coord[i][j] *= eVals[i];
				}
			}
			// set the new positions to the graph
			for (size_t i = 0; i < n; i++) {
				for (size_t d = 0; d < m_dimensionCount; d++) {
					G.Y[i * ((size_t)m_dimensionCount) + d] = coord[d][i];
				}
			}
			return choosenPivots;
		};

		//! Computes the eigen value decomposition based on power iteration.
		void eigenValueDecomposition(std::vector<std::vector<T>>& K, std::vector<std::vector<double>>& eVecs,
			std::vector<double>& eValues, bool timeSeries) {
			randomize(eVecs);
			const int p = K.size();
			T r = 0;
			for (int i = 0; i < m_dimensionCount; i++) {
				eValues[i] = normalize(eVecs[i]);
			}
			std::vector<std::vector<double>> tmpOld(m_dimensionCount);

			for (int i = 0; i < m_dimensionCount; i++) {
				tmpOld[i].resize(p);
			}
			int count = 0;
			int iter = 0;
			if (m_dimensionCount > 3) {
				EPSILON = 1 - 1e-5;
			}
			else {
				EPSILON = 1 - 1e-10;
			}

			while (r < EPSILON) {
				if (iter >= 10000) {
					std::cout << "Pmds did not converge after 10000 iterations, stopping early\n";
					break;
				}
				iter++;
				if (std::isnan(r) || isinf(r)) {
					// Throw arithmetic exception (Shouldn't occur
					// for DIMEMSION_COUNT = 2
					return;
				}
				// remember prev values
				for (int i = 0; i < m_dimensionCount; i++) {
					for (int j = 0; j < p; j++) {
						tmpOld[i][j] = eVecs[i][j];
						eVecs[i][j] = 0;
					}
				}
				// multiply matrices
				for (int i = 0; i < m_dimensionCount; i++) {
					for (int j = 0; j < p; j++) {
						for (int k = 0; k < p; k++) {
							eVecs[i][k] += K[j][k] * tmpOld[i][j];
						}
					}
				}

				// orthogonalize
				for (int i = 0; i < m_dimensionCount; i++) {
					for (int j = 0; j < i; j++) {
						T fac = prod(eVecs[j], eVecs[i]) / prod(eVecs[j], eVecs[j]);
						for (int k = 0; k < p; k++) {
							eVecs[i][k] -= fac * eVecs[j][k];
						}
					}
				}
				// normalize
				for (int i = 0; i < m_dimensionCount; i++) {
					eValues[i] = normalize(eVecs[i]);
				}
				r = 1;
				for (int i = 0; i < m_dimensionCount; i++) {
					// get absolute value (abs only defined for int)
					T tmp = prod(eVecs[i], tmpOld[i]);
					if (tmp < 0) {
						tmp *= -1;
					}
					if (r > tmp) {
						r = tmp;
					}
				}
				count++;
			}
			// Ensure time series stability
#ifdef PROCRUSTES
			if (timeSeries) {
				if (m_prevEvecs == nullptr) {
					m_prevEvecs = (double*)malloc(sizeof(double) * m_dimensionCount * p);
					for (int i = 0; i < m_dimensionCount; i++) {
						memcpy((void*)(m_prevEvecs + (i * p)), eVecs[i].data(), sizeof(double) * p);
					}
				}
				else {
					double* temp = (double*)malloc(sizeof(double) * m_dimensionCount * p);
					for (int i = 0; i < m_dimensionCount; i++) {
						memcpy((void*)(temp + (i * p)), eVecs[i].data(), sizeof(double) * p);
					}

					Procrustes::solve(temp, m_prevEvecs, m_dimensionCount, p);

					for (int i = 0; i < m_dimensionCount; i++) {
						memcpy(eVecs[i].data(), (void*)(temp + (i * p)), sizeof(double) * p);
					}
					free(temp);
				}
			}
#else
			if (m_prevEvecs.size() == 0) {
				// Save these eVecs
				m_prevEvecs.resize(eVecs.size());
				for (int i = 0; i < eVecs.size(); i++) {
					m_prevEvecs[i].resize(eVecs[i].size());
					memcpy(m_prevEvecs[i].data(), eVecs[i].data(), eVecs[i].size() * sizeof(double));
				}
			}
			else {
				// Allign current eVecs with previous
				for (int i = 0; i < eVecs.size(); i++) {
					auto res = prod(eVecs[i], m_prevEvecs[i]);
					if (res < 0) {
						for (int j = 0; j < eVecs[i].size(); j++) {
							eVecs[i][j] *= -1;
						}
					}
				}
			}
#endif
		};

		//! Computes the pivot distance matrix based on the maxmin strategy
		int* getPivotDistanceMatrix(Graph<T>& G, std::vector<std::vector<T>>& pivDistMatrix, bool* inAll) {
			const int n = G.nodeCount;

			// lower the number of pivots if necessary
			int numberOfPivots = m_numberOfPivots;
			if (numberOfPivots > n) {
				numberOfPivots = n;
			}
			int* choosenPivots = (int*)malloc(sizeof(int) * numberOfPivots);
			// number of pivots times n matrix used to store the graph distances
			pivDistMatrix.resize(numberOfPivots);
			int pivot = 0;
			std::vector<T> lowest;
			lowest.resize(n);
			for (int i = 0; i < numberOfPivots; i++) {
				pivDistMatrix[i].resize(n);
				G.getDistances(pivot, pivDistMatrix[i].data());

				if (i + 1 < numberOfPivots) {
					T highest = 0;
					if (i == 0) {
						std::memcpy(lowest.data(), pivDistMatrix[0].data(), n * sizeof(T));
						if (inAll == nullptr) {
							for (int j = 0; j < n; j++) {
								if (lowest[j] > highest) {
									highest = lowest[j];
									pivot = j;
								}
							}
						}
						else {
							choosenPivots[0] = G.nodeNames[0];
							for (int j = 0; j < n; j++) {
								if (lowest[j] > highest && inAll[G.nodeNames[j]]) {
									highest = lowest[j];
									pivot = j;
								}
							}
							choosenPivots[i + 1] = G.nodeNames[pivot];
						}
						continue;
					}
					if (inAll == nullptr) {
						for (int j = 0; j < n; j++) {
							if (pivDistMatrix[i][j] < lowest[j]) {
								lowest[j] = pivDistMatrix[i][j];
							}
							if (lowest[j] > highest) {
								highest = lowest[j];
								pivot = j;
							}
						}
					}
					else {
						choosenPivots[0] = G.nodeNames[0];
						for (int j = 0; j < n; j++) {
							if (!inAll[G.nodeNames[j]]) continue;
							if (pivDistMatrix[i][j] < lowest[j]) {
								lowest[j] = pivDistMatrix[i][j];
							}
							if (lowest[j] > highest) {
								highest = lowest[j];
								pivot = j;
							}
						}
						choosenPivots[i + 1] = G.nodeNames[pivot];
					}
				}
			}
			if (inAll == nullptr) {
				free(choosenPivots);
				return nullptr;
			}
			return choosenPivots;
		};

		//! Computes the pivot distance matrix based on the maxmin strategy
		void getPivotDistanceMatrix(Graph<T>& G, int* pivotPoints, std::vector<std::vector<T>>& pivDistMatrix) {
			const int n = G.nodeCount;

			// lower the number of pivots if necessary
			int numberOfPivots = m_numberOfPivots;
			if (numberOfPivots > n) {
				numberOfPivots = n;
			}
			// number of pivots times n matrix used to store the graph distances
			pivDistMatrix.resize(numberOfPivots);
			auto f = [&pivDistMatrix, &pivotPoints, &G, n](int begin, int end)
				{
					for (int i = begin; i < end; i++) {
						pivDistMatrix[i].resize(n);
						G.getDistances(G.namesToId[pivotPoints[i]], pivDistMatrix[i].data());
					}
				};
			Threadpool::divideWork(f, numberOfPivots);
		};

		//! Normalizes the vector \p x.
		template<typename P>
		T normalize(std::vector<P>& x) {
			T norm = sqrt(prod(x, x));
			if (norm != 0) {
				for (auto& elem : x) {
					elem /= norm;
				}
			}
			return norm;
		};

		//! Computes the product of two vectors \p x and \p y.
		template<typename P>
		T prod(const std::vector<P>& x, const std::vector<P>& y) {
			T result = 0;
			for (int i = 0; i < x.size(); i++) {
				result += x[i] * y[i];
			}
			return result;
		};

		//! Fills the given \p matrix with random doubles d 0 <= d <= 1.
		template<typename P>
		void randomize(std::vector<std::vector<P>>& matrix) {
			srand(SEED);
			for (auto& elem : matrix) {
				for (int j = 0; j < elem.size(); j++) {
					elem[j] = ((T)rand()) / RAND_MAX;
				}
			}
		};

		//! Computes the self product of \p d.
		template<typename P>
		void selfProduct(const std::vector<std::vector<P>>& d, std::vector<std::vector<P>>& result) {

			auto f = [&d, &result](int begin, int end) {
				for (int i = begin; i < end; i++) {
					for (int j = 0; j <= i; j++) {
						T sum = 0;
						for (int k = 0; k < d[0].size(); k++) {
							sum += d[i][k] * d[j][k];
						}
						result[i][j] = sum;
						result[j][i] = sum;
					}
				}
				};
			Threadpool::divideWork(f, d.size());

		};

		//! Computes the singular value decomposition of matrix \p K.
		void singularValueDecomposition(std::vector<std::vector<T>>& pivDistMatrix, std::vector<std::vector<double>>& eVecs,
			std::vector<double>& eVals, bool timeSeries) {
			const int size = pivDistMatrix.size();
			const int n = pivDistMatrix[0].size();
			std::vector<std::vector<T>> K(size);
			for (int i = 0; i < size; i++) {
				K[i].resize(size);
			}
			// calc C^TC
			selfProduct(pivDistMatrix, K);

			std::vector<std::vector<double>> tmp(m_dimensionCount);
			for (int i = 0; i < m_dimensionCount; i++) {
				tmp[i].resize(size);
			}

			eigenValueDecomposition(K, tmp, eVals, timeSeries);

			// C^Tx
			auto f = [this, &eVals, &eVecs, &pivDistMatrix, &tmp, n, size](int begin, int end) {
				for (int i = begin; i < end; i++) {
					eVals[i] = sqrt(eVals[i]);
					for (int j = 0; j < n; j++) { // node j
						eVecs[i][j] = 0;
					}
					for (int k = 0; k < size; k++) { // pivot k
						for (int j = 0; j < n; j++) { // node j
							eVecs[i][j] += pivDistMatrix[k][j] * tmp[i][k];
						}
					}

				}
				for (int i = begin; i < end; i++) {
					normalize(eVecs[i]);
				}
				};
			Threadpool::divideWork(f, m_dimensionCount);
		};
	};

}
