#pragma once

#include "Graph.h"
#include "Utils.h"

#include "Threading.h"

#define MAX_POINTS 100000000
#define MIN_POINTS 100

#define DISTANCE_THRESHOLD 0.00001

namespace NNPNet {
	class CalcStress {
	public:

		static double Calc(Graph<float>& g) {

			g.normalize();

			std::pair<float, float> max = { 0, 0 };

			int amountToCheck = MAX_POINTS / g.nodeCount;
			if (amountToCheck > g.nodeCount) {
				amountToCheck = g.nodeCount;
			}
			if (amountToCheck < MIN_POINTS) {
				amountToCheck = MIN_POINTS;
			}
			double inc = ((double)g.nodeCount) / ((double)amountToCheck);
			unsigned long long size = (unsigned long long)(amountToCheck) * (unsigned long long)g.nodeCount;
			std::pair<float, float>* allData = (std::pair<float, float>*)malloc(size * sizeof(std::pair<float, float>));
			float* _f_allData = (float*)allData;
			double top = 0;
			double bot = 0;
			auto res = Threadpool::divideWork<std::pair<std::pair<float, float>, std::pair<float, float>>>([inc, allData, &g, _f_allData](int begin, int end) -> std::pair<std::pair<float, float>, std::pair<float, float>> {
				double top = 0;
				double bot = 0;
				std::pair<float, float> max = { 0, 0 };
				for (int a = begin; a < end; a++) {
					long long i = (long long)(a * inc);
					g.getDistances(i, (float*)(allData + a * (long long)g.nodeCount));
					for (long long other = g.nodeCount - 1; other >= 0; other--) {
						if (other == a) {
							allData[a * (long long)g.nodeCount + other] = { 0, 0 };
							continue;
						}
						allData[(long long)a * (long long)g.nodeCount + (long long)other].first = _f_allData[((long long)2 * (long long)a * (long long)g.nodeCount) + (long long)other];
						std::pair<float, float> dif = { g.Y[i * 2] - g.Y[other * 2], g.Y[i * 2 + 1] - g.Y[other * 2 + 1] };
						allData[(long long)a * (long long)g.nodeCount + (long long)other].second = sqrt(dif.first * dif.first + dif.second * dif.second);
						if (allData[(long long)a * (long long)g.nodeCount + (long long)other].first > max.first) {
							max.first = allData[(long long)a * (long long)g.nodeCount + (long long)other].first;
						}
						if (allData[(long long)a * (long long)g.nodeCount + (long long)other].second > max.second) {
							max.second = allData[(long long)a * (long long)g.nodeCount + (long long)other].second;
						}
						if ((double)allData[(long long)a * (long long)g.nodeCount + (long long)other].first <= 0) {
							continue;
						}
						double temp = (double)allData[(long long)a * (long long)g.nodeCount + (long long)other].second / (double)allData[(long long)a * (long long)g.nodeCount + (long long)other].first;
						if (std::isnan(temp)) {
							continue;
						}
						bot += temp;
						top += temp * temp;
					}
				}
				return { max, std::pair{top, bot} };
				}, amountToCheck, [](std::pair<std::pair<float, float>, std::pair<float, float>> a, std::pair<std::pair<float, float>, std::pair<float, float>> b) ->std::pair<std::pair<float, float>, std::pair<float, float>>
					{
						std::pair<std::pair<float, float>, std::pair<float, float>> c = a;
						if (c.first.first < b.first.first) {
							c.first.first = b.first.first;
						}
						if (c.first.second < b.first.second) {
							c.first.second = b.first.second;
						}
						c.second.first += b.second.first;
						c.second.second += b.second.second;
						return c;
					});
				max = res.first;
				top = res.second.first;
				bot = res.second.second;

				float alpha = top / bot;

				double stress = 0;
				stress = getStress(g, amountToCheck, alpha, allData);
				std::cout << "Stress: " << stress << "\n";

				free(allData);
				return stress;
		}

		static double CalcFeatureStress(Graph<float>& g) {
			if (g.featureDimensions == 0) return -1;
			g.normalize();

			std::pair<float, float> max = { 0, 0 };

			int amountToCheck = MAX_POINTS / g.nodeCount;
			if (amountToCheck > g.nodeCount) {
				amountToCheck = g.nodeCount;
			}
			if (amountToCheck < MIN_POINTS) {
				amountToCheck = MIN_POINTS;
			}
			double inc = ((double)g.nodeCount) / ((double)amountToCheck);
			unsigned long long size = (unsigned long long)(amountToCheck) * (unsigned long long)g.nodeCount;
			std::pair<float, float>* allData = (std::pair<float, float>*)malloc(size * sizeof(std::pair<float, float>));
			float* _f_allData = (float*)allData;
			double top = 0;
			double bot = 0;
			auto res = Threadpool::divideWork<std::pair<std::pair<float, float>, std::pair<float, float>>>([inc, allData, &g, _f_allData](int begin, int end) -> std::pair<std::pair<float, float>, std::pair<float, float>> {
				double top = 0;
				double bot = 0;
				std::pair<float, float> max = { 0, 0 };
				for (int a = begin; a < end; a++) {
					long long i = (long long)(a * inc);
					g.addFeaturesToDistances(i, 1, (float*)(allData + a * (long long)g.nodeCount));
					for (long long other = g.nodeCount - 1; other >= 0; other--) {
						if (other == a) {
							allData[a * (long long)g.nodeCount + other] = { 0, 0 };
							continue;
						}
						allData[(long long)a * (long long)g.nodeCount + (long long)other].first = _f_allData[((long long)2 * (long long)a * (long long)g.nodeCount) + (long long)other];
						std::pair<float, float> dif = { g.Y[i * 2] - g.Y[other * 2], g.Y[i * 2 + 1] - g.Y[other * 2 + 1] };
						allData[(long long)a * (long long)g.nodeCount + (long long)other].second = sqrt(dif.first * dif.first + dif.second * dif.second);
						if (allData[(long long)a * (long long)g.nodeCount + (long long)other].first > max.first) {
							max.first = allData[(long long)a * (long long)g.nodeCount + (long long)other].first;
						}
						if (allData[(long long)a * (long long)g.nodeCount + (long long)other].second > max.second) {
							max.second = allData[(long long)a * (long long)g.nodeCount + (long long)other].second;
						}
						if ((double)allData[(long long)a * (long long)g.nodeCount + (long long)other].first <= 0) {
							continue;
						}
						double temp = (double)allData[(long long)a * (long long)g.nodeCount + (long long)other].second / (double)allData[(long long)a * (long long)g.nodeCount + (long long)other].first;
						if (std::isnan(temp)) {
							continue;
						}
						bot += temp;
						top += temp * temp;
					}
				}
				return { max, {top, bot} };
				}, amountToCheck, [](std::pair<std::pair<float, float>, std::pair<float, float>> a, std::pair<std::pair<float, float>, std::pair<float, float>> b) ->std::pair<std::pair<float, float>, std::pair<float, float>>
					{
						std::pair<std::pair<float, float>, std::pair<float, float>> c = a;
						if (c.first.first < b.first.first) {
							c.first.first = b.first.first;
						}
						if (c.first.second < b.first.second) {
							c.first.second = b.first.second;
						}
						c.second.first += b.second.first;
						c.second.second += b.second.second;
						return c;
					});
				max = res.first;
				top = res.second.first;
				bot = res.second.second;

				float alpha = top / bot;

				double stress = 0;
				stress = getStress(g, amountToCheck, alpha, allData);
				std::cout << "Feature Stress: " << stress << "\n";

				free(allData);
				return stress;
		}

	private:
		template<class T>
		static double getStress(Graph<T>& g, long long amountToCheck, double alpha, std::pair<float, float>* allData) {
			double stress = 0;
			for (long long i = 0; i < (long long)amountToCheck * (long long)g.nodeCount; i++) {
				if (allData[i].first < DISTANCE_THRESHOLD) continue;
				std::pair<float, float> stressVec = std::pair<float, float>{ allData[i].first*alpha, allData[i].second };

				double st = (stressVec.first - stressVec.second) / stressVec.first;
				stress += st * st;
			}
			stress /= (double)((long long)amountToCheck * (long long)g.nodeCount);
			return stress;
		}
	};
};