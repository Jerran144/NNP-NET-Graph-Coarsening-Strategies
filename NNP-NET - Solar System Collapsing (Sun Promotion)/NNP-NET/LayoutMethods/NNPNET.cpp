#include "NNPNET.h"

#include "../Threading.h"
#include "../Utils.h"
#include "PivotMDS.h"
#include "tsNET.h"

#include <algorithm>
#include <math.h>
#include <random>

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
  createFastSubnetwork(g, outG, nodes);
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
  for (int j = 0; j < inG.nodeCount; j++) {
    tempNodes[j] = j;
  }
  int currentNodeCount = inG.nodeCount;

  solarSystemIteration(inG, temp2, tempNodes, tempNodes2, target);

  if (temp2.nodeCount <= target && inG.nodeCount > target) {
    std::cout << temp2.nodeCount << "\n";
    // The very first iteration collapsed the graph below the target.
    // Use this iteration (below target) directly.
    outG = temp2;
    nodes.clear();
    for (int j : tempNodes2) {
      nodes.push_back(j);
    }
    outG.Y = (float *)malloc(sizeof(float) * outG.nodeCount * outG.outputDim);
    std::cout << "Number of subgraph points reached: " << outG.nodeCount
              << "\n";
    return;
  }

  currentNodeCount = temp2.nodeCount;
  bool currentIsFirst = false;
  float reduction = 0;

  while (currentNodeCount > target && reduction < 0.95) {
    std::cout << currentNodeCount << "\n";
    if (currentIsFirst) {
      solarSystemIteration(temp, temp2, tempNodes, tempNodes2, target);
      if (temp2.nodeCount <= target) {
        std::cout << temp2.nodeCount << "\n";
        currentIsFirst = false; // Use temp2 (the one below target)
        break;
      }
      currentNodeCount = temp2.nodeCount;
      currentIsFirst = false;
      if (temp.nodeCount > 0)
        reduction = ((float)temp2.nodeCount) / ((float)temp.nodeCount);
    } else {
      solarSystemIteration(temp2, temp, tempNodes2, tempNodes, target);
      if (temp.nodeCount <= target) {
        std::cout << temp.nodeCount << "\n";
        currentIsFirst = true; // Use temp (the one below target)
        break;
      }
      currentNodeCount = temp.nodeCount;
      currentIsFirst = true;
      if (temp2.nodeCount > 0)
        reduction = ((float)temp.nodeCount) / ((float)temp2.nodeCount);
    }
  }

  if (currentIsFirst) {
    if (currentNodeCount > target) {
      std::cout << "Did not collapse sufficiently (" << currentNodeCount
                << ")\n";
    }
    outG = temp;
    nodes.clear();
    for (int j : tempNodes) {
      nodes.push_back(j);
    }
  } else {
    if (currentNodeCount > target) {
      std::cout << "Did not collapse sufficiently (" << currentNodeCount
                << ")\n";
    }
    outG = temp2;
    nodes.clear();
    for (int j : tempNodes2) {
      nodes.push_back(j);
    }
  }
  outG.Y = (float *)malloc(sizeof(float) * outG.nodeCount * outG.outputDim);
  std::cout << "Number of subgraph points reached: " << outG.nodeCount << "\n";
}

void NNPNet::NNPNET::solarSystemIteration(Graph<float> &in, Graph<float> &out,
                                          std::vector<int> &inNodes,
                                          std::vector<int> &outNodes,
                                          int target) {
  out.edges.clear();
  out.nodeCount = 0;
  outNodes.clear();

  if (in.nodeCount <= 3) {
    out = in;
    outNodes = inNodes;
    return;
  }

  std::vector<int> celestial(in.nodeCount,
                             0); // 0=Undef, 1=Sun, 2=Planet, 3=Moon
  std::vector<int> orbitalCenter(in.nodeCount, -1);
  std::vector<float> distanceToOrbit(in.nodeCount, 0.0f);

  std::vector<int> suns;
  std::vector<int> candidates(in.nodeCount);
  for (int j = 0; j < in.nodeCount; j++)
    candidates[j] = j;

  static thread_local std::mt19937 rng(std::random_device{}());

  // Phase 1 + 2: Select Suns & Planets
  while (!candidates.empty()) {
    std::uniform_int_distribution<int> dist(0, candidates.size() - 1);
    int idx = dist(rng);
    int sun = candidates[idx];
    candidates[idx] = candidates.back();
    candidates.pop_back();

    if (celestial[sun] != 0)
      continue;

    bool hasForeignPlanet = false;
    for (Edge<float> &adj : in.edges[sun]) {
      if (celestial[adj.other] != 0) {
        hasForeignPlanet = true;
        break;
      }
    }
    if (hasForeignPlanet)
      continue;

    celestial[sun] = 1;
    suns.push_back(sun);
    for (Edge<float> &adj : in.edges[sun]) {
      celestial[adj.other] = 2;
      orbitalCenter[adj.other] = sun;
      distanceToOrbit[adj.other] = adj.weight;
    }
  }

  // Phase 3: Identify Moons
  for (int v = 0; v < in.nodeCount; v++) {
    if (celestial[v] == 0) {
      celestial[v] = 3;
      std::vector<Edge<float> *> planets;
      for (Edge<float> &adj : in.edges[v]) {
        if (celestial[adj.other] == 2) {
          planets.push_back(&adj);
        }
      }
      if (!planets.empty()) {
        std::uniform_int_distribution<int> dist(0, planets.size() - 1);
        int idx = dist(rng);
        orbitalCenter[v] = planets[idx]->other;
        distanceToOrbit[v] = planets[idx]->weight;
      } else {
        celestial[v] = 1;
        suns.push_back(v);
      }
    }
  }

  // Partial contraction: if full contraction would overshoot the target,
  // promote the farthest non-sun nodes back to suns so we hit the target
  // exactly.
  if (target > 0 && (int)suns.size() < target && in.nodeCount > target) {
    // Compute cumulative distance to sun for all non-sun nodes
    std::vector<float> distToSun(in.nodeCount, 0.0f);
    for (int i = 0; i < in.nodeCount; i++) {
      if (celestial[i] == 1)
        continue;
      float dist = 0;
      int curr = i;
      while (curr != -1 && celestial[curr] > 1) {
        dist += distanceToOrbit[curr];
        curr = orbitalCenter[curr];
      }
      distToSun[i] = dist;
    }

    // Collect all non-sun nodes with their distance to sun
    std::vector<std::pair<float, int>> nonSuns;
    for (int v = 0; v < in.nodeCount; v++) {
      if (celestial[v] != 1) {
        nonSuns.push_back({distToSun[v], v});
      }
    }

    // Sort by distance descending: promote farthest first
    std::sort(nonSuns.begin(), nonSuns.end(),
              [](const std::pair<float, int> &a,
                 const std::pair<float, int> &b) { return a.first > b.first; });

    int nodesNeeded = target - (int)suns.size();
    for (int i = 0; i < nodesNeeded && i < (int)nonSuns.size(); i++) {
      int v = nonSuns[i].second;
      celestial[v] = 1;
      suns.push_back(v);
    }

    std::cout << "Partial contraction: promoted " << nodesNeeded
              << " non-sun nodes to reach target " << target << "\n";
  }

  // Phase 4: Contraction
  out.nodeCount = suns.size();
  out.edges.resize(out.nodeCount);

  std::vector<int> nodeToSunIdx(in.nodeCount, -1);
  for (int j = 0; j < (int)suns.size(); j++) {
    int sun = suns[j];
    outNodes.push_back(inNodes[sun]);
    nodeToSunIdx[sun] = j;
  }

  // Map all non-sun nodes to their sun by walking the orbital chain.
  // This handles promoted nodes correctly: if a planet was promoted to sun,
  // its moons will walk up and find the promoted node as their sun.
  for (int v = 0; v < in.nodeCount; v++) {
    if (celestial[v] == 1)
      continue;
    int curr = v;
    while (curr != -1 && celestial[curr] != 1) {
      curr = orbitalCenter[curr];
    }
    if (curr != -1) {
      nodeToSunIdx[v] = nodeToSunIdx[curr];
    }
  }

  // Compute cumulative distance to assigned sun for edge weight calculation.
  // We use a separate vector to preserve the original hop distances in
  // distanceToOrbit (needed if promotion logic ran above).
  std::vector<float> cumulativeDist(in.nodeCount, 0.0f);
  for (int i = 0; i < in.nodeCount; i++) {
    if (celestial[i] == 1)
      continue;
    float dist = 0;
    int curr = i;
    while (curr != -1 && celestial[curr] > 1) {
      dist += distanceToOrbit[curr];
      curr = orbitalCenter[curr];
    }
    cumulativeDist[i] = dist;
  }

  std::vector<std::unordered_map<int, double>> newEdges(out.nodeCount);

  for (int j = 0; j < in.nodeCount; j++) {
    int metaI = nodeToSunIdx[j];
    if (metaI == -1)
      continue;
    for (Edge<float> &e : in.edges[j]) {
      int metaOther = nodeToSunIdx[e.other];
      if (metaOther == -1 || metaI == metaOther)
        continue;

      double dist = cumulativeDist[j] + cumulativeDist[e.other] + e.weight;
      if (newEdges[metaI].count(metaOther) == 0 ||
          newEdges[metaI][metaOther] > dist) {
        newEdges[metaI][metaOther] = dist;
      }
    }
  }

  for (int j = 0; j < out.nodeCount; j++) {
    for (auto &edge : newEdges[j]) {
      out.edges[j].push_back(Edge<float>(edge.first, edge.second));
    }
  }
  out.weighted = true;
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
