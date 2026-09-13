#pragma once

#include "Graph.h"

namespace NNPNet {

	class SmoothingFunctions {
	public:
		template <typename T>
		static void Laplacian(Graph<T>& g, const int passes = 1) {
			for (int _i = 0; _i < passes; _i++) {
				T* sY = (T*)malloc(sizeof(T) * g.outputDim * g.nodeCount);

				for (int j = 0; j < g.outputDim * g.nodeCount; j++) {
					sY[j] = 0;
				}
				
				int i = 0;
				for (auto& edges : g.edges) {
					double weights = 0;
					for (auto& e : edges) {
						for (int d = 0; d < g.outputDim; d++) {
							sY[i * g.outputDim + d] += g.Y[e.other * 2 + d] / e.weight;
						}
						weights += 1 / e.weight;
					}

					for (int d = 0; d < g.outputDim; d++) {
						sY[i * g.outputDim + d] /= weights;
					}

					i++;
				}
				free(g.Y);
				g.Y = sY;
			}
		}

	private:
		SmoothingFunctions() {};
	};

};
