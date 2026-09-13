#pragma once

#include "../Graph.h"

#include "PivotMDS.h"

namespace NNPNet {
class NNPNET {
public:
  NNPNET() {};

  void run(Graph<float> &g, Graph<double> *GT = nullptr,
           int embeddingSize = 50);
  void run(Graph<float> &g, int *pivotPoints, Graph<double> *GT = nullptr,
           int embeddingSize = 50);
  int *run(Graph<float> &g, bool *inAll, Graph<double> *GT = nullptr,
           int embeddingSize = 50);

  int subgraphPoints = 10000, perplexity = 40, pmdsPivots = 250,
      trainingEpochs = 40, batchSize = 64;
  bool pmdsEmbedding = true, fastSubgraph = true, gpu = false, useFloats = true;

  double theta = 0.25;
  double featureWeight = 0;

  bool fallbackFlag = false;
  std::string exportSubgraphPath = "";

private:
  float *createPivotEmbedding(Graph<float> &g, int pivots);

  float *createPMDSEmbedding(Graph<float> &g, int &dimensions, bool *inAll,
                             int **pivotPoints);

  void createSubnetwork(Graph<float> &inG, Graph<float> &outG,
                        std::vector<int> &nodes);

  void createFastSubnetwork(Graph<float> &inG, Graph<float> &outG,
                            std::vector<int> &nodes);
  void solarSystemIteration(Graph<float> &in, Graph<float> &out,
                            std::vector<int> &inNodes,
                            std::vector<int> &outNodes,
                            int target = 0);

  void trainNetwork(Graph<float> &g, Graph<double> *GT, float *embedding,
                    int embeddingDim);

  void multivariteSubgraph(Graph<float> &g, Graph<float> &subG,
                           std::vector<int> &nodes);
  void addMultivariateInformation(float *fullEmbeddingIn,
                                  float **fullEmbeddingOut,
                                  float *smallEmbeddingIn,
                                  float **smallEmbeddingOut, Graph<float> &g,
                                  std::vector<int> &nodesSubgraph,
                                  int &embeddingDim);

  void trainPlusInfer(float *smallEmbedding, int smallEmbeddingSize, double *gt,
                      float *fullEmbedding, int fullEmbeddingSize,
                      int embeddingDim, int outputDim, float *Y);
  void infer(float *fullEmbedding, int fullEmbeddingSize, int outputDim,
             float *Y);

  PivotMDS<float> pmds_f;
  PivotMDS<double> pmds_d;
};
}; // namespace NNPNet
