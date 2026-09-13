#include "NNPNET.h"

#include "../Threading.h"
#include "../Utils.h"
#include "PivotMDS.h"
#include "tsNET.h"

#include <math.h>

#include "pybind11/embed.h"
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace py::literals;

#define ALLOW_LESS_STRICT_MERGING
#define SET_ORDERING
// #define REVERSE_ORDER

void NNPNet::NNPNET::run(Graph<float> &g, Graph<double> *GT, int pivots) {
  float *embedding;
  if (pmdsEmbedding) {
    TIME(embedding = createPMDSEmbedding(g, pivots, nullptr, nullptr),
         "Create Embedding");
  } else {
    TIME(embedding = createPivotEmbedding(g, pivots), "Create Embedding");
  }

  trainNetwork(g, GT, embedding, pivots);
  free(embedding);
}

void NNPNet::NNPNET::run(Graph<float> &g, int *pivotPoints, Graph<double> *GT,
                         int pivots) {
  float *embedding;
  if (pmdsEmbedding) {
    TIME(embedding = createPMDSEmbedding(g, pivots, nullptr, &pivotPoints),
         "Create Embedding");
  } else {
    TIME(embedding = createPivotEmbedding(g, pivots), "Create Embedding");
  }

  // trainNetwork(g, GT, embedding, pivots);
  infer(embedding, g.nodeCount, g.outputDim, g.Y);
  free(embedding);
}

int *NNPNet::NNPNET::run(Graph<float> &g, bool *inAll, Graph<double> *GT,
                         int pivots) {
  float *embedding;
  int *pivotPoints;
  if (pmdsEmbedding) {
    TIME(embedding = createPMDSEmbedding(g, pivots, inAll, &pivotPoints),
         "Create Embedding");
  } else {
    TIME(embedding = createPivotEmbedding(g, pivots), "Create Embedding");
  }

  trainNetwork(g, GT, embedding, pivots);
  free(embedding);
  return pivotPoints;
}

float *NNPNet::NNPNET::createPivotEmbedding(Graph<float> &g, int pivots) {
  float *embedding =
      (float *)malloc(((size_t)g.nodeCount) * ((size_t)pivots) * sizeof(float));
  size_t n = g.nodeCount;
  size_t pivot = 0;
  float *lowest = (float *)malloc(sizeof(float) * n);
  for (size_t i = 0; i < pivots; i++) {
    g.getDistances(pivot, embedding + i * n);

    if (i + 1 < pivots) {
      float highest = 0;
      if (i == 0) {
        std::memcpy(lowest, embedding, n * sizeof(float));
        for (int j = 0; j < n; j++) {
          if (lowest[j] > highest) {
            highest = lowest[j];
            pivot = j;
          }
        }
        continue;
      }
      size_t p_i = i * n;
      for (int j = 0; j < n; j++) {
        if (embedding[p_i] < lowest[j]) {
          lowest[j] = embedding[p_i];
        }
        if (lowest[j] > highest) {
          highest = lowest[j];
          pivot = j;
        }
        p_i++;
      }
    }
  }
  float biggest = 0;
  for (size_t i = 0; i < g.nodeCount; i++) {
    for (size_t j = 0; j < pivots; j++) {
      if (embedding[j * ((size_t)g.nodeCount) + i] > biggest) {
        biggest = embedding[j * ((size_t)g.nodeCount) + i];
      }
    }
  }
  for (size_t i = 0; i < g.nodeCount; i++) {
    for (size_t j = 0; j < pivots; j++) {
      embedding[j * ((size_t)g.nodeCount) + i] /= biggest;
    }
  }

  return embedding;
}

float *NNPNet::NNPNET::createPMDSEmbedding(Graph<float> &g, int &dimensions,
                                           bool *inAll, int **pivotPoints) {
  float *embedding = (float *)malloc(((size_t)g.nodeCount) *
                                     ((size_t)dimensions) * sizeof(float));
  size_t n = g.nodeCount;
  float *backupY = g.Y;
  size_t backupDim = g.outputDim;
  g.Y = embedding;
  g.outputDim = dimensions;

  if (useFloats) {

    pmds_f.setNumberOfPivots(pmdsPivots);
    if (inAll != nullptr) {
      (*pivotPoints) = pmds_f.call(g, inAll);
    } else if (pivotPoints != nullptr) {
      pmds_f.call(g, *pivotPoints);
    } else {
      pmds_f.call(g);
    }
  } else {
    Graph<double> g_d(g);

    pmds_d.setNumberOfPivots(pmdsPivots);
    if (inAll != nullptr) {
      (*pivotPoints) = pmds_d.call(g_d, inAll);
    } else if (pivotPoints != nullptr) {
      pmds_d.call(g_d, *pivotPoints);
    } else {
      pmds_d.call(g_d);
    }
    for (int i = 0; i < ((size_t)g.nodeCount) * ((size_t)dimensions); i++) {
      g.Y[i] = (float)g_d.Y[i];
    }
  }

  bool hasNaN = false;
  for (size_t i = 0; i < ((size_t)g.nodeCount) * ((size_t)dimensions); i++) {
    if (std::isnan(embedding[i])) {
      hasNaN = true;
      break;
    }
  }

  if (!hasNaN)
    g.normalize();
  g.Y = backupY;
  g.outputDim = backupDim;

  if (hasNaN) {
    std::cout
        << "PMDS embedding contained NaN's, using pivots as the embedding\n";
    free(embedding);
    return createPivotEmbedding(g, dimensions);
  }
  return embedding;
}

void NNPNet::NNPNET::createSubnetwork(Graph<float> &g, Graph<float> &outG,
                                      std::vector<int> &nodes) {
  int targetPoints = subgraphPoints;
  outG.Y = (float *)malloc(targetPoints * g.outputDim * sizeof(float));
  float *embedding = (float *)malloc(g.nodeCount * sizeof(float));
  int n = g.nodeCount;
  int pivot = 0;
  float *lowest = (float *)malloc(sizeof(float) * n);
  outG.edges.resize(targetPoints);
  for (int i = 0; i < targetPoints; i++) {
    if (i == 0) {
      g.getDistances(pivot, embedding);
    } else {
      g.getDistancesWithBound(pivot, embedding, lowest);
    }
    nodes.push_back(pivot);

    // Next pivot point
    if (i + 1 < targetPoints) {
      float highest = 0;
      if (i == 0) {
        std::memcpy(lowest, embedding, n * sizeof(float));
        for (int j = 0; j < n; j++) {
          if (lowest[j] > highest) {
            highest = lowest[j];
            pivot = j;
          }
        }
        continue;
      }
      for (int j = 0; j < n; j++) {
        if (embedding[j] != -1 && embedding[j] < lowest[j]) {
          lowest[j] = embedding[j];
        }
        if (lowest[j] > highest) {
          highest = lowest[j];
          pivot = j;
        }
      }
    }

    // Add edges
    float l = 9999999999;
    for (int j = 0; j < i; j++) {
      float dist = embedding[nodes[j]];
      if (dist == -1) {
        // Node was pruned; fall back to the stored minimum distance
        dist = lowest[nodes[j]];
      }
      if (dist < l) {
        l = dist;
      }
    }

    for (int j = 0; j < i; j++) {
      float dist = embedding[nodes[j]];
      if (dist == -1) {
        dist = lowest[nodes[j]];
      }
      outG.edges[i].push_back(Edge(j, dist));
      outG.edges[j].push_back(Edge(i, dist));
    }
  }
  free(lowest);
  free(embedding);
  outG.nodeCount = targetPoints;
}

void NNPNet::NNPNET::createFastSubnetwork(Graph<float> &inG, Graph<float> &outG,
                                          std::vector<int> &nodes) {
  int target = subgraphPoints;
  Graph<float> temp(inG.outputDim);
  Graph<float> temp2(inG.outputDim);
  outG.weighted = true;
  temp.weighted = true;
  temp2.weighted = true;

  std::vector<int> tempNodes;
  std::vector<int> tempNodes2;
  // fill temp nodes with all numbers
  tempNodes.resize(inG.nodeCount);
  for (int i = 0; i < inG.nodeCount; i++) {
    tempNodes[i] = i;
  }
  int currentNodeCount = inG.nodeCount;
  int startPos = 0;
  startPos = fastSubnetworkIteration(inG, temp2, tempNodes, tempNodes2, target,
                                     startPos);
  currentNodeCount = temp2.nodeCount;
  bool currentIsFirst = false;
  float reduction = 0;
  while (currentNodeCount > target && reduction < 0.95) {
    std::cout << currentNodeCount << "\n";
    if (currentIsFirst) {
      startPos = fastSubnetworkIteration(temp, temp2, tempNodes, tempNodes2,
                                         target, startPos);
      currentNodeCount = temp2.nodeCount;
      currentIsFirst = false;
      reduction = ((float)temp2.nodeCount) / ((float)temp.nodeCount);
    } else {
      startPos = fastSubnetworkIteration(temp2, temp, tempNodes2, tempNodes,
                                         target, startPos);
      currentNodeCount = temp.nodeCount;
      currentIsFirst = true;
      reduction = ((float)temp.nodeCount) / ((float)temp2.nodeCount);
    }
  }
  if (currentIsFirst) {
    if (currentNodeCount > target) {
      std::cout << "Does not collapse nicelly, reverting to pivot method to "
                   "reduced graph\n";
      fallbackFlag = true;
      temp.weighted = true;
      createSubnetwork(temp, outG, nodes);
    } else {
      outG = temp;
      for (int i : tempNodes) {
        nodes.push_back(i);
      }
    }
  } else {
    if (currentNodeCount > target) {
      std::cout << "Does not collapse nicelly, reverting to pivot method to "
                   "reduced graph\n";
      fallbackFlag = true;
      temp2.weighted = true;
      createSubnetwork(temp2, outG, nodes);
    } else {
      outG = temp2;
      for (int i : tempNodes2) {
        nodes.push_back(i);
      }
    }
  }
  outG.Y = (float *)malloc(sizeof(float) * outG.nodeCount * outG.outputDim);
  std::cout << "Number of subgraph points reached: " << outG.nodeCount << "\n";
}

void NNPNet::NNPNET::getFastSubnetworkOrder(
    Graph<float> &g, std::vector<std::pair<int, int>> &list) {
  // Fill list
  list.resize(g.nodeCount);
  for (int i = 0; i < g.nodeCount; i++) {
    list[i].first = g.edges[i].size();
    list[i].second = i;
  }
  std::vector<std::pair<int, int>> list2;
  list2.resize(g.nodeCount);
  // Radix sort
  int counts[256];
  for (int i = 0; i < 2; i++) {
    // Reset counts
    for (int j = 0; j < 256; j++)
      counts[j] = 0;
    // Count
    int off = i * 2;
    for (int j = 0; j < g.nodeCount; j++) {
      counts[*((unsigned char *)(list.data() + j) + off)]++;
    }
    // Set offsets
    int curr = 0;
#ifdef REVERSE_ORDER
    for (int j = 255; j >= 0; j--) {
#else
    for (int j = 0; j < 256; j++) {
#endif
      int next = curr + counts[j];
      counts[j] = curr;
      curr = next;
    }
    // Move into new list
    for (int j = 0; j < g.nodeCount; j++) {
      list2[counts[*((unsigned char *)(list.data() + j) + off)]++] = list[j];
    }

    // Again, but from list2 -> list
    // Reset counts
    for (int j = 0; j < 256; j++)
      counts[j] = 0;
    // Count
    off = i * 2 + 1;
    for (int j = 0; j < g.nodeCount; j++) {
      counts[*((unsigned char *)(list2.data() + j) + off)]++;
    }
    // Set offsets
    curr = 0;
#ifdef REVERSE_ORDER
    for (int j = 255; j >= 0; j--) {
#else
    for (int j = 0; j < 256; j++) {
#endif
      int next = curr + counts[j];
      counts[j] = curr;
      curr = next;
    }
    // Move into new list
    for (int j = 0; j < g.nodeCount; j++) {
      list[counts[*((unsigned char *)(list2.data() + j) + off)]++] = list2[j];
    }
  }
}

int NNPNet::NNPNET::fastSubnetworkIteration(Graph<float> &in, Graph<float> &out,
                                            std::vector<int> &inNodes,
                                            std::vector<int> &outNodes,
                                            int target, int startPos) {
  out.edges.clear();
  out.nodeCount = 0;
  outNodes.clear();
  int *partOfNode = (int *)malloc(sizeof(int) * in.nodeCount);
  double *distanceFromCenter = (double *)malloc(sizeof(double) * in.nodeCount);
  startPos -= 1;
  if (startPos == -1) {
    startPos += in.nodeCount;
  }
  // Set all nodes as not part of a new node
  for (int i = 0; i < in.nodeCount; i++) {
    partOfNode[i] = -1;
    distanceFromCenter[i] = 0;
  }
  // Create clusters
  std::cout << in.nodeCount << " -> ";
  int currentNodeCount = in.nodeCount;
#ifdef SET_ORDERING
  std::vector<std::pair<int, int>> order;
  getFastSubnetworkOrder(in, order);
  for (int j = 0; j < in.nodeCount; j++) {
#ifdef REVERSE_ORDER
    int left = currentNodeCount - target + 10;
    while (order[j].first > left && j < in.nodeCount - 1) {
      j++;
    }
#endif
    int i = order[j].second;
#else
  for (int i = (startPos + 1) % in.nodeCount; i != startPos;
       i = (i + 1) % in.nodeCount) {
#endif
    if (partOfNode[i] == -1) {
#ifndef ALLOW_LESS_STRICT_MERGING
      bool neighborsFree = true;
      // Only if none of the nodes that would be included
      // are already part of another cluster
      for (Edge &e : in.edges[i]) {
        if (partOfNode[e.other] != -1) {
          neighborsFree = false;
          break;
        }
      }
#endif

      // Add it to the list of nodes
#ifdef ALLOW_LESS_STRICT_MERGING
      {
#else
      if (neighborsFree) {
#endif
        partOfNode[i] = out.nodeCount;
        int count = 0;
        for (Edge<float> &e : in.edges[i]) {
          if (partOfNode[e.other] == -1 && e.other != i) {
            partOfNode[e.other] = out.nodeCount;
            distanceFromCenter[e.other] = e.weight;
            count++;
          }
        }
        outNodes.push_back(inNodes[i]);
        currentNodeCount -= count;
        out.nodeCount++;
        if (currentNodeCount <= target) {
          break;
        }
      }
    }
  }
  // Add everything that has not an assigned node yet
  for (int i = 0; i < in.nodeCount; i++) {
    if (partOfNode[i] == -1) {
      outNodes.push_back(inNodes[i]);
      partOfNode[i] = out.nodeCount;
      out.nodeCount++;
    }
  }
  int nextStart = (startPos + out.nodeCount / 4) % out.nodeCount;
  // Edges
  std::vector<std::unordered_map<int, double>> edges;
  edges.resize(out.nodeCount);
  for (int i = 0; i < in.nodeCount; i++) {
    for (Edge e : in.edges[i]) {
      if (partOfNode[e.other] == partOfNode[i])
        continue;
      double dist =
          distanceFromCenter[i] + distanceFromCenter[e.other] + e.weight;
      if (edges[partOfNode[i]].count(partOfNode[e.other]) == 0 ||
          edges[partOfNode[i]][partOfNode[e.other]] > dist) {
        edges[partOfNode[i]][partOfNode[e.other]] = dist;
      }
    }
  }
  // Copy edges into the right format
  out.edges.resize(out.nodeCount);
  for (int i = 0; i < out.nodeCount; i++) {
    for (auto &edge : edges[i]) {
      out.edges[i].push_back(Edge<float>(edge.first, edge.second));
    }
  }

  free(partOfNode);
  return nextStart;
}

void NNPNet::NNPNET::trainNetwork(Graph<float> &g, Graph<double> *GT,
                                  float *embedding, int embeddingDim) {
  Graph<float> subG(g.outputDim);
  std::vector<int> nodes;

  if (GT != nullptr) {
    // Convert to output labels
    // First train on a subset for faster convergence
    if (g.nodeCount > subgraphPoints * 1.5) {
      if (fastSubgraph) {
        TIME(createFastSubnetwork(g, subG, nodes), "Create subgraph");
      } else {
        TIME(createSubnetwork(g, subG, nodes), "Create subgraph");
      }

      float *smallEmbedding =
          (float *)malloc(subG.nodeCount * embeddingDim * sizeof(float));
      for (int i = 0; i < subG.nodeCount; i++) {
        for (int d = 0; d < embeddingDim; d++) {
          smallEmbedding[i * embeddingDim + d] =
              embedding[((size_t)nodes[i]) * ((size_t)embeddingDim) +
                        (size_t)d];
        }
      }
      double *outputLabels =
          (double *)malloc(subG.nodeCount * GT->outputDim * sizeof(double));
      for (int i = 0; i < subG.nodeCount; i++) {
        for (int d = 0; d < GT->outputDim; d++) {
          outputLabels[i * GT->outputDim + d] =
              GT->Y[nodes[i] * GT->outputDim + d];
        }
      }

      TIME(trainPlusInfer(smallEmbedding, subG.nodeCount, outputLabels,
                          embedding, g.nodeCount, embeddingDim, g.outputDim,
                          g.Y),
           "Train and Inference");
      free(smallEmbedding);
      free(outputLabels);
      return;
    }
    TIME(trainPlusInfer(embedding, g.nodeCount, GT->Y, embedding, g.nodeCount,
                        embeddingDim, g.outputDim, g.Y),
         "Train and Inference");

  } else if (g.nodeCount > subgraphPoints * 1.5) {
    if (fastSubgraph) {
      TIME(createFastSubnetwork(g, subG, nodes), "Create fast subgraph");
    } else {
      TIME(createSubnetwork(g, subG, nodes), "Create subgraph");
    }
    multivariteSubgraph(g, subG, nodes);

    TSNET<double> tsnet;
    tsnet.perp = perplexity;
    tsnet.featureWeight = featureWeight;
    Graph<double> subD(subG);
    TIME(tsnet.tsNETStar(subD, theta), "creating ground truth");

    subD.normalize();
    if (exportSubgraphPath != "") {
      subD.saveToVNA(exportSubgraphPath);
    }

    // Convert to output labels
    float *smallEmbedding =
        (float *)malloc(subG.nodeCount * embeddingDim * sizeof(float));
    for (int i = 0; i < subG.nodeCount; i++) {
      for (int d = 0; d < embeddingDim; d++) {
        smallEmbedding[i * embeddingDim + d] =
            embedding[((size_t)nodes[i]) * ((size_t)embeddingDim) + (size_t)d];
      }
    }

    if (featureWeight > 0 && g.featureDimensions > 0) {
      float *_fullEmbedding, *_smallEmbedding;
      addMultivariateInformation(embedding, &_fullEmbedding, smallEmbedding,
                                 &_smallEmbedding, g, nodes, embeddingDim);
      TIME(trainPlusInfer(_smallEmbedding, subG.nodeCount, subD.Y,
                          _fullEmbedding, g.nodeCount, embeddingDim,
                          g.outputDim, g.Y),
           "Train and Inference");
      free(_fullEmbedding);
      free(_smallEmbedding);
    } else {
      TIME(trainPlusInfer(smallEmbedding, subG.nodeCount, subD.Y, embedding,
                          g.nodeCount, embeddingDim, g.outputDim, g.Y),
           "Train and Inference");
    }
    free(smallEmbedding);
  } else {
    TSNET<double> tsnet;
    tsnet.perp = perplexity;
    tsnet.featureWeight = featureWeight;
    Graph<double> gd(g);
    TIME(tsnet.tsNETStar(gd, theta), "creating ground truth");
    gd.normalize();
    if (exportSubgraphPath != "") {
      gd.saveToVNA(exportSubgraphPath);
    }

    // Convert to output labels
    if (featureWeight > 0 && g.featureDimensions > 0) {
      float *_fullEmbedding;
      addMultivariateInformation(embedding, &_fullEmbedding, nullptr, nullptr,
                                 g, nodes, embeddingDim);
      TIME(trainPlusInfer(_fullEmbedding, g.nodeCount, gd.Y, _fullEmbedding,
                          g.nodeCount, embeddingDim, g.outputDim, g.Y),
           "Train and Inference");
      free(_fullEmbedding);
    } else {
      TIME(trainPlusInfer(embedding, g.nodeCount, gd.Y, embedding, g.nodeCount,
                          embeddingDim, g.outputDim, g.Y),
           "Train and Inference");
    }
  }
}

void NNPNet::NNPNET::multivariteSubgraph(Graph<float> &g, Graph<float> &subG,
                                         std::vector<int> &nodes) {
  if (featureWeight <= 0 || g.featureDimensions <= 0)
    return;

  subG.features =
      (float *)malloc(g.featureDimensions * subG.nodeCount * sizeof(float));
  subG.featureDimensions = g.featureDimensions;
  for (int i = 0; i < subG.nodeCount; i++) {
    memcpy(subG.features + (i * g.featureDimensions),
           g.features + (nodes[i] * g.featureDimensions),
           sizeof(float) * g.featureDimensions);
  }
}

void NNPNet::NNPNET::addMultivariateInformation(
    float *fullEmbeddingIn, float **fullEmbeddingOut, float *smallEmbeddingIn,
    float **smallEmbeddingOut, Graph<float> &g, std::vector<int> &nodesSubgraph,
    int &embeddingDim) {
  if (g.featureDimensions == 0) {
    (*fullEmbeddingOut) = fullEmbeddingIn;
    (*smallEmbeddingOut) = smallEmbeddingIn;
    return;
  }

  int newEmbeddingSize = embeddingDim + g.featureDimensions;
  (*fullEmbeddingOut) =
      (float *)malloc(sizeof(float) * newEmbeddingSize * g.nodeCount);

  for (int i = 0; i < g.nodeCount; i++) {
    memcpy(*fullEmbeddingOut + (i * newEmbeddingSize),
           fullEmbeddingIn + (i * embeddingDim), sizeof(float) * embeddingDim);
    memcpy(*fullEmbeddingOut + (i * newEmbeddingSize + embeddingDim),
           g.features + (i * g.featureDimensions),
           sizeof(float) * g.featureDimensions);
  }
  if (smallEmbeddingIn != nullptr) {
    (*smallEmbeddingOut) = (float *)malloc(sizeof(float) * newEmbeddingSize *
                                           nodesSubgraph.size());
    for (int i = 0; i < nodesSubgraph.size(); i++) {
      memcpy(*smallEmbeddingOut + (i * newEmbeddingSize),
             smallEmbeddingIn + (i * embeddingDim),
             sizeof(float) * embeddingDim);
      memcpy(*smallEmbeddingOut + (i * newEmbeddingSize + embeddingDim),
             g.features + (nodesSubgraph[i] * g.featureDimensions),
             sizeof(float) * g.featureDimensions);
    }
  }

  embeddingDim = newEmbeddingSize;
}

static float *_gt = nullptr;
static float *_fullEmbedding = nullptr;
static float *_smallEmbedding = nullptr;
static bool first = true;
static void *loc = nullptr;
static int _embeddingDim;

// PYBIND11_EMBEDDED_MODULE(getLists, m) {
// 	m.def("getSmallEmbedding", [](size_t smallEmbeddingSize, size_t
// embeddingDim) { 		return py::array_t<float>({ smallEmbeddingSize,
// embeddingDim
// }, _smallEmbedding);
// 		});
// 	m.def("getGt", [](size_t smallEmbeddingSize, size_t outputDim) {
// 		return py::array_t<float>({ smallEmbeddingSize, outputDim },
// _gt);
// 		});
// 	m.def("getFullEmbedding", [](size_t fullEmbeddingSize, size_t
// embeddingDim) { 		return py::array_t<float>({ fullEmbeddingSize,
// embeddingDim
// }, _fullEmbedding);
// 		});
// }

void NNPNet::NNPNET::trainPlusInfer(float *smallEmbedding,
                                    int smallEmbeddingSize, double *gt,
                                    float *fullEmbedding, int fullEmbeddingSize,
                                    int embeddingDim, int outputDim, float *Y) {
  float *fgt = (float *)malloc(smallEmbeddingSize * outputDim * sizeof(float));
  for (int i = 0; i < smallEmbeddingSize * outputDim; i++) {
    fgt[i] = (float)gt[i];
  }
  _gt = fgt;
  _fullEmbedding = fullEmbedding;
  _smallEmbedding = smallEmbedding;
  _embeddingDim = embeddingDim;
  // Else the loss function gives an error
  smallEmbeddingSize -= (smallEmbeddingSize % 64);

  // Call the python script
  if (loc != nullptr)
    delete (py::dict *)loc;
  auto l = new py::dict(
      "smallEmbeddingSize"_a = smallEmbeddingSize,
      "fullEmbeddingSize"_a = fullEmbeddingSize,
      "embeddingDim"_a = embeddingDim, "trainEpochs"_a = trainingEpochs,
      "batchSize"_a = batchSize, "outputDim"_a = outputDim,
      "smallEmbedding"_a = py::array_t<float>(
          {smallEmbeddingSize, embeddingDim}, _smallEmbedding),
      "gt"_a = py::array_t<float>({smallEmbeddingSize, outputDim}, _gt),
      "fullEmbedding"_a = py::array_t<float>({fullEmbeddingSize, embeddingDim},
                                             _fullEmbedding));
  loc = l;

  py::exec(
      R"(
import keras
from keras import layers
import tensorflow as tf
)" +
          (gpu ? std::string()
               : std::string("tf.config.set_visible_devices([], 'GPU')\n")) +
          R"(import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[embeddingDim], batch_size=batchSize),
    tf.keras.layers.Dense(256, activation="leaky_relu"),
    tf.keras.layers.Dense(512, activation="leaky_relu"),
    tf.keras.layers.Dense(256, activation="leaky_relu"),
    tf.keras.layers.Dense(outputDim)
    ])
model.compile(optimizer='Adam',
              loss=tf.keras.losses.MeanSquaredError())
    
model.fit(smallEmbedding, gt, epochs=trainEpochs, batch_size=batchSize)
    
outPredictions = model.predict(fullEmbedding, batch_size=4096)
)",
      py::globals(), (*(py::dict *)loc));

  float *out =
      (float *)(*(py::dict *)loc)["outPredictions"].cast<py::array>().data();

  memcpy(Y, out, fullEmbeddingSize * outputDim * sizeof(float));

  free(fgt);
}

void NNPNet::NNPNET::infer(float *fullEmbedding, int fullEmbeddingSize,
                           int outputDim, float *Y) {
  _fullEmbedding = fullEmbedding;
  (*(py::dict *)loc)["fullEmbeddingSize"] = fullEmbeddingSize;
  (*(py::dict *)loc)["fullEmbedding"] =
      py::array_t<float>({fullEmbeddingSize, _embeddingDim}, _fullEmbedding);
  py::exec(R"(
outPredictions = model.predict(fullEmbedding, batch_size=4096)
)",
           py::globals(), (*(py::dict *)loc));
  float *out =
      (float *)(*(py::dict *)loc)["outPredictions"].cast<py::array>().data();

  memcpy(Y, out, fullEmbeddingSize * outputDim * sizeof(float));
}
