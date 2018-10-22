/***************************************************************************
 *
 * Authors:    David Strelak (davidstrelak@gmail.com)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#include "reconstruction_adapt_cuda/movie_alignment_correlation_gpu.h"

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::defineParams() {
    AProgMovieAlignmentCorrelation<T>::defineParams();
    this->addParamsLine("  [--device <dev=0>]                 : GPU device to use. 0th by default");
    this->addParamsLine("  [--storage <fn=\"\">]              : Path to file that can be used to store results of the benchmark");
    this->addExampleLine(
                "xmipp_cuda_movie_alignment_correlation -i movie.xmd --oaligned alignedMovie.stk --oavg alignedMicrograph.mrc --device 0");
    this->addSeeAlsoLine("xmipp_movie_alignment_correlation");
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::show() {
    AProgMovieAlignmentCorrelation<T>::show();
    std::cout << "Device:              " << device << " " << getUUID(device) << std::endl;
    std::cout << "Benchmark storage    " << (storage.empty() ? "Default" : storage) << std::endl;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::readParams() {
    AProgMovieAlignmentCorrelation<T>::readParams();
    device = this->getIntParam("--device");
    storage = this->getParam("--storage");
}

template<typename T>
std::pair<T,T> getMaxBoundary(std::vector<std::pair<T,T> > shifts) {
	T minX = std::numeric_limits<T>::max();
	T maxX = std::numeric_limits<T>::min();
	T minY = std::numeric_limits<T>::max();
	T maxY = std::numeric_limits<T>::min();
	for (const auto& s : shifts) {
		minX = std::min(std::floor(s.first), minX);
		maxX = std::max(std::ceil(s.first), maxX);
		minY = std::min(std::floor(s.second), minY);
		maxY = std::max(std::ceil(s.second), maxY);
	}
	std::cout << minX << " " << maxX << " " << minY << " " << maxY << std::endl;
	return std::pair<T,T>(std::abs(maxX - minX), std::abs(maxY - minY));
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::getPatches(size_t idx, size_t idy,
		T* data, std::pair<T,T>& border, std::vector<std::pair<T,T> >& shifts, T* result) {
	size_t n = shifts.size();
	size_t xFirst = border.first + (idx * patchSizeX);
	size_t yFirst = border.second + (idy * patchSizeY);
	for (size_t i = 0; i < n; ++i) {
		size_t frameOffset = i * inputOptSizeX * inputOptSizeY;
		size_t patchOffset = i * patchSizeX * patchSizeY;
		int xShift = shifts.at(i).first;
		int yShift = shifts.at(i).second;
//		printf("%d %d \n", xShift, yShift);
		for (size_t y = 0; y < patchSizeY; ++y) {
			size_t srcY = (yFirst + y) - yShift; // assuming shift is smaller than offset
			if ((srcY >=0) && (srcY < inputOptSizeY)) {
				size_t srcIndex = (frameOffset + (srcY * inputOptSizeX) + xFirst) - xShift;
				size_t destIndex = patchOffset + y * patchSizeX;
				memcpy(result + destIndex, data + srcIndex, patchSizeX * sizeof(T));
			} else {
				printf("ERROR!!!!!!!!!!!\n");
			}
		}
	}
}

template<typename T>
void computeShifts(int device, size_t scaledMaxShift, size_t N, std::complex<T>* data,
		size_t fftX, size_t x, size_t y, size_t bufferImgs, size_t batch,
		const Matrix1D<T>& bX, const Matrix1D<T>& bY, const Matrix2D<T>& A) {
	setDevice(device);
//	// since we are using different size of FFT, we need to scale results to
//	// 'expected' size
	T localSizeFactorX = 1;
//			/ (croppedOptSizeX / (T) inputOptSizeX);
	T localSizeFactorY = 1;
//			/ (croppedOptSizeY / (T) inputOptSizeY);
//	size_t scaledMaxShift = std::floor((this->maxShift * this->sizeFactor) / localSizeFactorX);

	T* correlations;
	size_t centerSize = std::ceil(scaledMaxShift * 2 + 1);
	computeCorrelations(centerSize, N, data, fftX,
			x, y, bufferImgs,
			batch, correlations);

	Image<T> corrs(centerSize, centerSize, 1, (N*(N-1))/2);
	corrs.data.data = correlations;
	corrs.write("correlations.vol");
	corrs.data.data = NULL;

	int idx = 0;
	MultidimArray<T> Mcorr(centerSize, centerSize);
	for (size_t i = 0; i < N - 1; ++i) {
		for (size_t j = i + 1; j < N; ++j) {
			size_t offset = idx * centerSize * centerSize;
			Mcorr.data = correlations + offset;
			Mcorr.setXmippOrigin();
			bestShift(Mcorr, bX(idx), bY(idx), NULL, scaledMaxShift);
			bX(idx) *= localSizeFactorX; // scale to expected size
			bY(idx) *= localSizeFactorY;
			if (true)
//				std::cerr << "Frame " << i << " to Frame "
//						<< j  << " -> ("
//						<< bX(idx) << ","
//						<< bY(idx) << ")" << std::endl;
			for (int ij = i; ij < j; ij++)
				A(idx, ij) = 1;

			idx++;
		}
	}
	Mcorr.data = NULL;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::applyLocalShiftsComputeAverage(
        T *data, size_t x, size_t y,
        std::vector<std::pair<T, T> > globalShifts,
        Matrix1D<T>& shiftX, Matrix1D<T>& shiftY,
        int N, size_t counter, int device, int bestIref) {
    // Apply shifts and compute average
    GeoShiftTransformer<T> transformer;
    size_t size = globalShifts.size();
    Image<T> croppedFrame(x, y);
    Image<T> shiftedFrame(x, y);
    Image<T> averageMicrograph(x, y);
    for (size_t n = 0; n < size; ++n)
    {
    	T totShiftX;// = shiftX(n);
//				+ globalShifts.at(n).first; // no global shift yet, incoming data are 'aligned'
    	T totShiftY;// = shiftY(n);
//				+ globalShifts.at(n).second; // no global shift yet, incoming data are 'aligned'
    	this->computeTotalShift(bestIref, n, shiftX, shiftY, totShiftX,
    			totShiftY);
    	totShiftX = globalShifts.at(n).first + totShiftX;
    	totShiftY = globalShifts.at(n).second + totShiftY;
    	croppedFrame.data.data = data + (n * x * y);
//		std::cout << n << " shiftX=" << totShiftX << " shiftY="
//                    << totShiftY << std::endl;
		transformer.initLazy(x,
				y, 1, device);
		transformer.applyShift(shiftedFrame(), croppedFrame(), totShiftX, totShiftY);
		if (n == 0)
			averageMicrograph() = shiftedFrame();
		else
			averageMicrograph() += shiftedFrame();
	}
    croppedFrame.data.data = NULL;
    averageMicrograph.write("avg" + std::to_string(counter) + ".vol");
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::applyLocalShiftsComputeAverage(
        T *data, size_t x, size_t y, size_t xSize, size_t ySize,
        std::map<std::tuple<size_t,size_t, size_t>, std::pair<T,T>> shifts,
        int N, size_t counter, int device) {
    // Apply shifts and compute average
    GeoShiftTransformer<T> transformer;
    size_t size = N;
    Image<T> croppedFrame(xSize, ySize);
    Image<T> shiftedFrame(xSize, ySize);
    Image<T> averageMicrograph(xSize, ySize);
    for (size_t n = 0; n < size; ++n)
    {
    	T totShiftX = shifts.at(std::make_tuple(x,y,n)).first;
    	T totShiftY = shifts.find(std::make_tuple(x,y,n))->second.second;
    	croppedFrame.data.data = data + (n * xSize * ySize);
//		std::cout << n << " shiftX=" << totShiftX << " shiftY="
//                    << totShiftY << std::endl;
		transformer.initLazy(xSize,	ySize, 1, device);
		transformer.applyShift(shiftedFrame(), croppedFrame(), totShiftX, totShiftY);
		if (n == 0)
			averageMicrograph() = shiftedFrame();
		else
			averageMicrograph() += shiftedFrame();
	}
    croppedFrame.data.data = NULL;
    averageMicrograph.write("kontrola" + std::to_string(counter) + ".vol");
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getTotalShift(
		const std::tuple<size_t,size_t, size_t>& tile,
		const std::vector<std::pair<T,T>>& globalShifts,
		const Matrix1D<T>& shiftX, const Matrix1D<T>& shiftY,
		int bestIref) {
	size_t t = std::get<2>(tile);
//	static_assert(t < globalShifts.size());
	T x, y;
	this->computeTotalShift(bestIref, t,
			shiftX, shiftY, x, y);
	return std::make_pair(globalShifts.at(t).first + x,
			globalShifts.at(t).second + y);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::computeLocalShifts(MetaData& movie,
            Image<T>& dark, Image<T>& gain, int bestIref) {
	std::vector<std::pair<T,T> >shifts;
	std::vector<std::pair<T,T>>zeroShifts;
	int n = -1;
	bool cropInput = (this->yDRcorner != -1);
	int noOfImgs = this->nlast - this->nfirst + 1;
	FOR_ALL_OBJECTS_IN_METADATA(movie)
	{
		n++;
		if (n >= this->nfirst && n <= this->nlast) {
			std::pair<T,T> s;
			movie.getValue(MDL_SHIFT_X, s.first, __iter.objId);
			movie.getValue(MDL_SHIFT_Y, s.second, __iter.objId);
			s.first = std::round(s.first);
			s.second = std::round(s.second);
			shifts.push_back(s);
			zeroShifts.emplace_back(0,0);
		}
	}
//	zeroShifts.resize(shifts.size());
	T* data = loadToRAM(movie, noOfImgs, dark, gain, cropInput);
	std::cout << "compute local shifts" << std::endl;
//	for (const auto shift : shifts) {
//		std::cout << shift.first << " " << shift.second << std::endl;
//	}
	localShiftBorder = getMaxBoundary(shifts);
	patchSizeX = patchSizeY = 400;

	T* tmp = new T[noOfImgs * patchSizeX
	            * std::max(patchSizeX, (((patchSizeX/2)+1)*2) * 2)]();
	size_t noOfPatchesX = 10;
	size_t noOfPatchesY = 10;
	std::map<std::tuple<size_t,size_t, size_t>, std::pair<T,T>> tilesShifts;
	auto k = std::make_tuple(77,88,99);
	auto v = std::make_pair(11,22);
	tilesShifts.emplace(k, v);
	bool eq = tilesShifts.begin()->first == std::make_tuple(77,88,99);
	std::cout << std::boolalpha << "udelej novy funguje:" << eq << std::endl;
	std::cout << std::boolalpha << "find funguje:" << (tilesShifts.find(std::make_tuple(77,88,99)) == tilesShifts.begin()) << std::endl;
	std::cout << std::boolalpha << "[] funguje:" << (tilesShifts[std::make_tuple(77,88,99)].first == 11) << std::endl;
//	std::cout << tilesShifts.begin()->first
//	Image<T> patch(patchSizeX, patchSizeY, 1, noOfImgs, data);
	for (int y=0; y < noOfPatchesY;++y ) {
		for (int x = 0; x < noOfPatchesX; ++x) {
			getPatches(x, y, data, localShiftBorder, shifts, tmp);
		    // scale and transform to FFT on GPU
		    performFFTAndScale<T>(tmp, noOfImgs, patchSizeX,
		            patchSizeY, 50, patchSizeX/2+1,
		            patchSizeY, nullptr);
		    size_t N = noOfImgs;
		    Matrix2D<T> A(N * (N - 1) / 2, N - 1);
			Matrix1D<T> bX(N * (N - 1) / 2), bY(N * (N - 1) / 2);
			printf("Patch %d\n",y*noOfPatchesX+x);
			::computeShifts(device, this->maxShift, noOfImgs, (std::complex<T>*)tmp,
					patchSizeX/2+1, patchSizeX, patchSizeY,  60,  10,
					bX, bY, A);
			Matrix1D<T> shiftX, shiftY;
			this->solveEquationSystem(bX, bY, A, shiftX, shiftY);
			// Choose reference image as the minimax of shifts
//			int bestIref = this->findReferenceImage(N, shiftX, shiftY);
			auto func = [&](int t) {
				T lx, ly;
				this->computeTotalShift(bestIref, t,
						shiftX, shiftY, lx, ly);
				T gx = shifts.at(t).first;
				T gy = shifts.at(t).second;
				T tx = gx - lx;
				T ty = gy - ly;
//				printf("local shift: %f %f", lx, ly);
//				printf("\tglobal shift: %f %f", gx, gy);
//				printf("\ttotal shift: %f %f\n", tx, ty);
//				printf("\treference: %d\n", bestIref);
				return std::make_pair(tx,ty);
			};
			for (size_t t = 0; t < N; ++t) {
				tilesShifts.emplace(std::make_tuple(x,y,t), func(t));
			}

			//// DEBUG
//			getPatches(x, y, data, localShiftBorder, zeroShifts, tmp);
//			Image<T> zkouska(patchSizeX, patchSizeY,1, noOfImgs);
//			zkouska.data.data = tmp;
//			zkouska.write("zkouska" + std::to_string(y*N+x) + ".vol");
//			zkouska.data.data = NULL;
//			applyLocalShiftsComputeAverage(
//			        tmp, x, y,patchSizeX, patchSizeY,
//			        tilesShifts,
//			        noOfImgs, (y*noOfPatchesX+x), device);

//			break;

			// Choose reference image as the minimax of shifts
//			int bestIref = findReferenceImage(N, shiftX, shiftY);
//			patch.write("test" + std::to_string(y*10+x) + ".vol");
		}
//		break;
	}
	int L = 4;
	int Lt = 3;
	int noOfPatchesXY = noOfPatchesX*noOfPatchesY;
	Matrix2D<T>A(noOfPatchesXY*noOfImgs, (L+2)*(L+2)*(Lt+2));
	Matrix1D<T>bX(noOfPatchesXY*noOfImgs);
	Matrix1D<T>bY(noOfPatchesXY*noOfImgs);
	T hX = inputOptSizeX / (T)(L-1);
	T hY = inputOptSizeY / (T)(L-1);
	T hT = noOfImgs / (T)(Lt-1);
//	printf("hX hY: %f %f %f\n", hX, hY, hT);
	for (int t = 0 ; t < noOfImgs; ++t) {
		int tileIdxT = t;
		int tileCenterT = tileIdxT * 1 + 0 + 0;
//		printf("%d\n", t);
		for (int i = 0; i < noOfPatchesXY; ++i) {
			int tileIdxX = i%noOfPatchesX;
			int tileIdxY = i/noOfPatchesX;

			int tileCenterX = tileIdxX * patchSizeX + patchSizeX/2 + localShiftBorder.first;
			int tileCenterY = tileIdxY * patchSizeY + patchSizeY/2 +localShiftBorder.second;

//			printf("\ttile %d %d %d (%d %d %d):\n", tileIdxX, tileIdxY, tileIdxT, tileCenterX, tileCenterY, tileCenterT);

			for (int j = 0; j < (Lt+2)*(L+2)*(L+2); ++j) {
				int controlIdxT = j/((L+2)*(L+2))-1;
				int XY=j%((L+2)*(L+2));
     			int controlIdxY = (XY/(L+2)) -1;
				int controlIdxX = (XY%(L+2)) -1;
				// note: if control point is not in the tile vicinity, val == 0 and can be skipped
				T val = Bspline03((tileCenterX / (T)hX) - controlIdxX) *
						Bspline03((tileCenterY / (T)hY) - controlIdxY) *
						Bspline03((tileCenterT / (T)hT) - controlIdxT);
				//A.mdata[(t*noOfPatchesXY*A.mdimy)+(i*noOfPatchesXY+j)] = val;
				MAT_ELEM(A,t*noOfPatchesXY + i,j) = val;
//				printf("\t\t i=%d,control point %d %d %d = %f\n", t*noOfPatchesXY + i, controlIdxX, controlIdxY, controlIdxT, val);

			}
			VEC_ELEM(bX,t*noOfPatchesXY + i) = tilesShifts[std::make_tuple(tileIdxX, tileIdxY, t)].first;
			VEC_ELEM(bY,t*noOfPatchesXY + i) = tilesShifts[std::make_tuple(tileIdxX, tileIdxY, t)].second;
//			printf("\tB = %f %f\n\n", tilesShifts[std::make_tuple(tileIdxX, tileIdxY, t)].first,
//					tilesShifts[std::make_tuple(tileIdxX, tileIdxY, t)].second);
		}
	}

	Matrix1D<T> coefsX, coefsY;
	this->solveEquationSystem(bX, bY, A, coefsX, coefsY);
	std::cout << coefsX << std::endl;
	std::cout << coefsY << std::endl;
//	for (auto &r : tilesShifts) {
//		std::cout << std::get<0>(r.first) << " "
//				<< std::get<1>(r.first) << " "
//				<< std::get<2>(r.first) << ": "
//				<< std::get<0>(r.second) << " "
//				<< std::get<1>(r.second) << std::endl;
//	}
}




template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
        const MetaData& movie, const Image<T>& dark, const Image<T>& gain,
        Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
        size_t& N) {
    // Apply shifts and compute average
    Image<T> frame, croppedFrame, reducedFrame, shiftedFrame;
    Matrix1D<T> shift(2);
    FileName fnFrame;
    int j = 0;
    int n = 0;
    Ninitial = N = 0;
    GeoShiftTransformer<T> transformer;
    FOR_ALL_OBJECTS_IN_METADATA(movie)
    {
        if (n >= this->nfirstSum && n <= this->nlastSum) {
            movie.getValue(MDL_IMAGE, fnFrame, __iter.objId);
            movie.getValue(MDL_SHIFT_X, XX(shift), __iter.objId);
            movie.getValue(MDL_SHIFT_Y, YY(shift), __iter.objId);

            std::cout << fnFrame << " shiftX=" << XX(shift) << " shiftY="
                    << YY(shift) << std::endl;
            frame.read(fnFrame);
            if (XSIZE(dark()) > 0)
                frame() -= dark();
            if (XSIZE(gain()) > 0)
                frame() *= gain();
            if (this->yDRcorner != -1)
                frame().window(croppedFrame(), this->yLTcorner, this->xLTcorner,
                        this->yDRcorner, this->xDRcorner);
            else
                croppedFrame() = frame();
            if (this->bin > 0) {
                // FIXME add templates to respective functions/classes to avoid type casting
                Image<double> croppedFrameDouble;
                Image<double> reducedFrameDouble;
                typeCast(croppedFrame(), croppedFrameDouble());

                scaleToSizeFourier(1, floor(YSIZE(croppedFrame()) / this->bin),
                        floor(XSIZE(croppedFrame()) / this->bin),
                        croppedFrameDouble(), reducedFrameDouble());

                typeCast(reducedFrameDouble(), reducedFrame());

                shift /= this->bin;
                croppedFrame() = reducedFrame();
            }

            if (this->fnInitialAvg != "") {
                if (j == 0)
                    initialMic() = croppedFrame();
                else
                    initialMic() += croppedFrame();
                Ninitial++;
            }

            if (this->fnAligned != "" || this->fnAvg != "") {
                if (this->outsideMode == OUTSIDE_WRAP) {
//                    Matrix2D<T> tmp;
//                    translation2DMatrix(shift, tmp, true);
                    transformer.initLazy(croppedFrame().xdim,
                            croppedFrame().ydim, 1, device);
                    transformer.applyShift(shiftedFrame(), croppedFrame(), XX(shift), YY(shift));
//                    transformer.applyGeometry(this->BsplineOrder,
//                            shiftedFrame(), croppedFrame(), tmp, IS_INV, WRAP);
                } else if (this->outsideMode == OUTSIDE_VALUE)
                    translate(this->BsplineOrder, shiftedFrame(),
                            croppedFrame(), shift, DONT_WRAP,
                            this->outsideValue);
                else
                    translate(this->BsplineOrder, shiftedFrame(),
                            croppedFrame(), shift, DONT_WRAP,
                            (T) croppedFrame().computeAvg());
                if (this->fnAligned != "")
                    shiftedFrame.write(this->fnAligned, j + 1, true,
                            WRITE_REPLACE);
                if (this->fnAvg != "") {
                    if (j == 0)
                        averageMicrograph() = shiftedFrame();
                    else
                        averageMicrograph() += shiftedFrame();
                    N++;
                }
            }
            j++;
        }
        n++;
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::loadFrame(const MetaData& movie,
        size_t objId, bool crop, Image<T>& out) {
    FileName fnFrame;
    movie.getValue(MDL_IMAGE, fnFrame, objId);
    if (crop) {
        Image<T> tmp;
        tmp.read(fnFrame);
        tmp().window(out(), this->yLTcorner, this->xLTcorner, this->yDRcorner,
                this->xDRcorner);
    } else {
        out.read(fnFrame);
    }
}

template<typename T>
int ProgMovieAlignmentCorrelationGPU<T>::getMaxFilterSize(Image<T> &frame) {
    size_t maxXPow2 = std::ceil(log(frame.data.xdim) / log(2));
    size_t maxX = std::pow(2, maxXPow2);
    size_t maxFFTX = maxX / 2 + 1;
    size_t maxYPow2 = std::ceil(log(frame.data.ydim) / log(2));
    size_t maxY = std::pow(2, maxYPow2);
    size_t bytes = maxFFTX * maxY * sizeof(T);
    return bytes / (1024 * 1024);
}

template<typename T>
T* ProgMovieAlignmentCorrelationGPU<T>::loadToRAM(const MetaData& movie,
        int noOfImgs, const Image<T>& dark, const Image<T>& gain,
        bool cropInput) {
    // allocate enough memory for the images. Since it will be reused, it has to be big
    // enough to store either all FFTs or all input images
    T* imgs = new T[noOfImgs * inputOptSizeY
            * std::max(inputOptSizeX, inputOptSizeFFTX * 2)]();
    Image<T> frame;

    int movieImgIndex = -1;
    FOR_ALL_OBJECTS_IN_METADATA(movie)
    {
        // update variables
        movieImgIndex++;
        if (movieImgIndex < this->nfirst)
            continue;
        if (movieImgIndex > this->nlast)
            break;

        // load image
        loadFrame(movie, __iter.objId, cropInput, frame);
        if (XSIZE(dark()) > 0)
            frame() -= dark();
        if (XSIZE(gain()) > 0)
            frame() *= gain();

        // copy line by line, adding offset at the end of each line
        // result is the same image, padded in the X and Y dimensions
        T* dest = imgs
                + ((movieImgIndex - this->nfirst) * inputOptSizeX
                        * inputOptSizeY); // points to first float in the image
        for (size_t i = 0; i < inputOptSizeY; ++i) {
            memcpy(dest + (inputOptSizeX * i),
                    frame.data.data + i * frame.data.xdim,
                    inputOptSizeX * sizeof(T));
        }
    }
    return imgs;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::setSizes(Image<T> &frame,
        int noOfImgs) {

    std::string UUID = getUUID(device);

    int maxFilterSize = getMaxFilterSize(frame);
    size_t availableMemMB = getFreeMem(device);
    correlationBufferSizeMB = availableMemMB / 3; // divide available memory to 3 parts (2 buffers + 1 FFT)

    if (! getStoredSizes(frame, noOfImgs, UUID)) {
        runBenchmark(frame, noOfImgs, UUID);
        storeSizes(frame, noOfImgs, UUID);
    }

    T corrSizeMB = ((size_t) croppedOptSizeFFTX * croppedOptSizeY
            * sizeof(std::complex<T>)) / (1024 * 1024.);
    correlationBufferImgs = std::ceil(correlationBufferSizeMB / corrSizeMB);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::runBenchmark(Image<T> &frame,
        int noOfImgs, std::string &uuid) {
    // get best sizes
    int maxFilterSize = getMaxFilterSize(frame);
    if (this->verbose)
        std::cerr << "Benchmarking cuFFT ..." << std::endl;

    size_t noOfCorrelations = (noOfImgs * (noOfImgs - 1)) / 2;

    // we also need enough memory for filter
    getBestFFTSize(noOfImgs, frame.data.xdim, frame.data.ydim, inputOptBatchSize,
            true,
            inputOptSizeX, inputOptSizeY, maxFilterSize, this->verbose, device,
            frame().xdim == frame().ydim, 10); // allow max 10% change

    inputOptSizeFFTX = inputOptSizeX / 2 + 1;

    getBestFFTSize(noOfCorrelations, this->newXdim, this->newYdim,
            croppedOptBatchSize, false, croppedOptSizeX, croppedOptSizeY,
            correlationBufferSizeMB * 2, this->verbose, device,
            this->newXdim == this->newYdim, 10);

    croppedOptSizeFFTX = croppedOptSizeX / 2 + 1;
}


template<typename T>
bool ProgMovieAlignmentCorrelationGPU<T>::getStoredSizes(Image<T> &frame,
        int noOfImgs, std::string &uuid) {
    bool res = true;
    size_t neededMem;
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, inputOptSizeXStr, frame, noOfImgs, true), inputOptSizeX);
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, inputOptSizeYStr, frame, noOfImgs, true), inputOptSizeY);
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, inputOptBatchSizeStr, frame, noOfImgs, true), inputOptBatchSize);
    inputOptSizeFFTX =  inputOptSizeX / 2 + 1;
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, availableMemoryStr, frame, noOfImgs, true), neededMem);
    res = res && neededMem <= getFreeMem(device);

    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, croppedOptSizeXStr, this->newXdim, this->newYdim, noOfImgs, false), croppedOptSizeX);
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, croppedOptSizeYStr, this->newXdim, this->newYdim, noOfImgs, false), croppedOptSizeY);
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, croppedOptBatchSizeStr, this->newXdim, this->newYdim, noOfImgs, false),
        croppedOptBatchSize);
    croppedOptSizeFFTX =  croppedOptSizeX / 2 + 1;
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, availableMemoryStr, this->newXdim, this->newYdim, noOfImgs, false), neededMem);
    res = res && neededMem <= getFreeMem(device);

    return res;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::storeSizes(Image<T> &frame,
        int noOfImgs, std::string &uuid) {
    UserSettings::get(storage).insert(*this,
        getKey(uuid, inputOptSizeXStr, frame, noOfImgs, true), inputOptSizeX);
    UserSettings::get(storage).insert(*this,
        getKey(uuid, inputOptSizeYStr, frame, noOfImgs, true), inputOptSizeY);
    UserSettings::get(storage).insert(*this,
        getKey(uuid, inputOptBatchSizeStr, frame, noOfImgs, true),
        inputOptBatchSize);
    UserSettings::get(storage).insert(*this,
        getKey(uuid, availableMemoryStr, frame, noOfImgs, true), getFreeMem(device));

    UserSettings::get(storage).insert(*this,
        getKey(uuid, croppedOptSizeXStr, this->newXdim, this->newYdim, noOfImgs, false),
        croppedOptSizeX);
    UserSettings::get(storage).insert(*this,
        getKey(uuid, croppedOptSizeYStr, this->newXdim, this->newYdim, noOfImgs, false),
        croppedOptSizeY);
    UserSettings::get(storage).insert(*this,
        getKey(uuid, croppedOptBatchSizeStr,
        this->newXdim, this->newYdim, noOfImgs, false),
        croppedOptBatchSize);
    UserSettings::get(storage).insert(*this,
        getKey(uuid, availableMemoryStr, this->newXdim, this->newYdim, noOfImgs, false),
        getFreeMem(device));
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testFFT() {

    double delta = 0.00001;
    size_t x, y;
    x = y = 2304;
    size_t order = 10000;

    srand(42);

    Image<double> inputDouble(x, y); // keep sync with values
    Image<float> inputFloat(x, y); // keep sync with values
    size_t pixels = inputDouble.data.xdim * inputDouble.data.ydim;
    for (size_t y = 0; y < inputDouble.data.ydim; ++y) {
        for (size_t x = 0; x < inputDouble.data.xdim; ++x) {
            size_t index = y * inputDouble.data.xdim + x;
            double value = rand() / (RAND_MAX / 2000.);
            inputDouble.data.data[index] = value;
            inputFloat.data.data[index] = (float) value;
        }
    }

    // CPU part

    MultidimArray<std::complex<double> > tmpFFTCpu;
    FourierTransformer transformer;

    transformer.FourierTransform(inputDouble(), tmpFFTCpu, true);

    // store results to drive
    Image<double> fftCPU(tmpFFTCpu.xdim, tmpFFTCpu.ydim);
    size_t fftPixels = fftCPU.data.yxdim;
    for (size_t i = 0; i < fftPixels; i++) {
        fftCPU.data.data[i] = tmpFFTCpu.data[i].real();
    }
    fftCPU.write("testFFTCpu.vol");

    // GPU part

    GpuMultidimArrayAtGpu<float> gpuIn(inputFloat.data.xdim,
            inputFloat.data.ydim);
    gpuIn.copyToGpu(inputFloat.data.data);
    GpuMultidimArrayAtGpu<std::complex<float> > gpuFFT;
    mycufftHandle handle;
    gpuIn.fft(gpuFFT, handle);

    fftPixels = gpuFFT.yxdim;
    std::complex<float>* tmpFFTGpu = new std::complex<float>[fftPixels];
    gpuFFT.copyToCpu(tmpFFTGpu);

    // store results to drive
    Image<float> fftGPU(gpuFFT.Xdim, gpuFFT.Ydim);
    float norm = inputFloat.data.yxdim;
    for (size_t i = 0; i < fftPixels; i++) {
        fftGPU.data.data[i] = tmpFFTGpu[i].real() / norm;
    }
    fftGPU.write("testFFTGpu.vol");

    ////////////////////////////////////////

    if (fftCPU.data.xdim != fftGPU.data.xdim) {
        printf("wrong size: X cpu %lu X gpu %lu\n", fftCPU.data.xdim,
                fftGPU.data.xdim);
    }
    if (fftCPU.data.ydim != fftGPU.data.ydim) {
        printf("wrong size: Y cpu %lu Y gpu %lu\n", fftCPU.data.xdim,
                fftGPU.data.xdim);
    }

    for (size_t i = 0; i < fftCPU.data.yxdim; ++i) {
        float cpuReal = tmpFFTCpu.data[i].real();
        float cpuImag = tmpFFTCpu.data[i].imag();
        float gpuReal = tmpFFTGpu[i].real() / norm;
        float gpuImag = tmpFFTGpu[i].imag() / norm;
        if ((std::abs(cpuReal - gpuReal) > delta)
                || (std::abs(cpuImag - gpuImag) > delta)) {
            printf("ERROR FFT: %lu cpu (%f, %f) gpu (%f, %f)\n", i, cpuReal,
                    cpuImag, gpuReal, gpuImag);
        }
    }

    delete[] tmpFFTGpu;

}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testFilterAndScale() {
    double delta = 0.00001;
    size_t xIn, yIn, xOut, yOut, xOutFFT;
    xIn = yIn = 4096;
    xOut = yOut = 2275;
    xOutFFT = xOut / 2 + 1;

    size_t fftPixels = xOutFFT * yOut;
    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[fftPixels];
    float* filter = new float[fftPixels];
    for (size_t i = 0; i < fftPixels; ++i) {
        filter[i] = (rand() * 100) / (float) RAND_MAX;
    }

    srand(42);

    Image<double> inputDouble(xIn, yIn); // keep sync with values
    Image<float> inputFloat(xIn, yIn); // keep sync with values
    size_t pixels = inputDouble.data.xdim * inputDouble.data.ydim;
    for (size_t y = 0; y < inputDouble.data.ydim; ++y) {
        for (size_t x = 0; x < inputDouble.data.xdim; ++x) {
            size_t index = y * inputDouble.data.xdim + x;
            double value = rand() > (RAND_MAX / 2) ? -1 : 1; // ((int)(1000 * (double)rand() / (RAND_MAX))) / 1000.f;
            inputDouble.data.data[index] = value;
            inputFloat.data.data[index] = (float) value;
        }
    }
//	inputDouble(0,0) = 1;
//	inputFloat(0,0) = 1;
    Image<double> outputDouble(xOut, yOut);
    Image<double> reducedFrame;

    // CPU part

    scaleToSizeFourier(1, yOut, xOut, inputDouble(), reducedFrame());
//	inputDouble().printStats();
//	printf("\n");
//	reducedFrame().printStats();
//	printf("\n");
    // Now do the Fourier transform and filter
    MultidimArray<std::complex<double> > *tmpFFTCpuOut = new MultidimArray<
            std::complex<double> >;
    MultidimArray<std::complex<double> > *tmpFFTCpuOutFull = new MultidimArray<
            std::complex<double> >;
    FourierTransformer transformer;

    transformer.FourierTransform(inputDouble(), *tmpFFTCpuOutFull);
//	std::cout << *tmpFFTCpuOutFull<< std::endl;

    transformer.FourierTransform(reducedFrame(), *tmpFFTCpuOut, true);
    for (size_t nn = 0; nn < fftPixels; ++nn) {
        double wlpf = filter[nn];
        DIRECT_MULTIDIM_ELEM(*tmpFFTCpuOut,nn) *= wlpf;
    }

    // store results to drive
    Image<double> fftCPU(tmpFFTCpuOut->xdim, tmpFFTCpuOut->ydim);
    fftPixels = tmpFFTCpuOut->yxdim;
    for (size_t i = 0; i < fftPixels; i++) {
        fftCPU.data.data[i] = tmpFFTCpuOut->data[i].real();
        if (fftCPU.data.data[i] > 10)
            fftCPU.data.data[i] = 0;
    }
    fftCPU.write("testFFTCpuScaledFiltered.vol");

    // GPU part

    float* d_filter = loadToGPU(filter, fftPixels);

    GpuMultidimArrayAtGpu<float> gpuIn(inputFloat.data.xdim,
            inputFloat.data.ydim);
    gpuIn.copyToGpu(inputFloat.data.data);
    GpuMultidimArrayAtGpu<std::complex<float> > gpuFFT;
    mycufftHandle handle;

//    processInput(gpuIn, gpuFFT, handle, xIn, yIn, 1, xOutFFT, yOut, d_filter,
//            tmpFFTGpuOut); // FIXME test

    // store results to drive
    Image<float> fftGPU(xOutFFT, yOut);
    float norm = inputFloat.data.yxdim;
    for (size_t i = 0; i < fftPixels; i++) {
        fftGPU.data.data[i] = tmpFFTGpuOut[i].real() / norm;
        if (fftGPU.data.data[i] > 10)
            fftGPU.data.data[i] = 0;
    }
    fftGPU.write("testFFTGpuScaledFiltered.vol");

    ////////////////////////////////////////

    if (fftCPU.data.xdim != fftGPU.data.xdim) {
        printf("wrong size: X cpu %lu X gpu %lu\n", fftCPU.data.xdim,
                fftGPU.data.xdim);
    }
    if (fftCPU.data.ydim != fftGPU.data.ydim) {
        printf("wrong size: Y cpu %lu Y gpu %lu\n", fftCPU.data.xdim,
                fftGPU.data.xdim);
    }
    if (tmpFFTCpuOut->ydim != yOut) {
        printf("wrong size tmpFFTCpuOut: Y cpu %lu Y gpu %lu\n",
                tmpFFTCpuOut->ydim, yOut);
    }
    if (tmpFFTCpuOut->xdim != xOutFFT) {
        printf("wrong size tmpFFTCpuOut: X cpu %lu X gpu %lu\n",
                tmpFFTCpuOut->xdim, xOutFFT);
    }

    for (size_t i = 0; i < fftCPU.data.yxdim; ++i) {
        float cpuReal = tmpFFTCpuOut->data[i].real();
        float cpuImag = tmpFFTCpuOut->data[i].imag();
        float gpuReal = tmpFFTGpuOut[i].real() / norm;
        float gpuImag = tmpFFTGpuOut[i].imag() / norm;
        if ((std::abs(cpuReal - gpuReal) > delta)
                || (std::abs(cpuImag - gpuImag) > delta)) {
            printf("ERROR FILTER: %lu cpu (%f, %f) gpu (%f, %f)\n", i, cpuReal,
                    cpuImag, gpuReal, gpuImag);
        }
    }
    delete[] tmpFFTGpuOut;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingGpuOO() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
    xIn = yIn = 9;
    xOut = yOut = 5;
    xOutFFT = xOut / 2 + 1; // == 3
    xInFFT = xIn / 2 + 1; // == 5

    std::complex<float>* tmpFFTGpuIn = new std::complex<float>[yIn * xInFFT];
    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[yOut * xOutFFT];
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < yIn; ++y) {
        for (size_t x = 0; x < xInFFT; ++x) {
            size_t index = y * xInFFT + x;
            tmpFFTGpuIn[index] = std::complex<float>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);

    tmpFFTCpuOutExpected[3] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 2);

    tmpFFTCpuOutExpected[6] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[7] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 2);

    tmpFFTCpuOutExpected[9] = std::complex<double>(7, 0);
    tmpFFTCpuOutExpected[10] = std::complex<double>(7, 1);
    tmpFFTCpuOutExpected[11] = std::complex<double>(7, 2);

    tmpFFTCpuOutExpected[12] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(8, 2);

//    applyFilterAndCrop<float>(tmpFFTGpuIn, tmpFFTGpuOut, 1, xInFFT, yIn,
//            xOutFFT, yOut, NULL); // FIXME test

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTGpuOut[i].real();
        float cpuImag = tmpFFTGpuOut[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE GPU OO: %lu gpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingGpuEO() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
    xIn = yIn = 10;
    xOut = yOut = 5;
    xOutFFT = xOut / 2 + 1; // == 3
    xInFFT = xIn / 2 + 1; // == 6

    std::complex<float>* tmpFFTGpuIn = new std::complex<float>[yIn * xInFFT];
    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[yOut * xOutFFT];
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < yIn; ++y) {
        for (size_t x = 0; x < xInFFT; ++x) {
            size_t index = y * xInFFT + x;
            tmpFFTGpuIn[index] = std::complex<float>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);

    tmpFFTCpuOutExpected[3] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 2);

    tmpFFTCpuOutExpected[6] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[7] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 2);

    tmpFFTCpuOutExpected[9] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[10] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[11] = std::complex<double>(8, 2);

    tmpFFTCpuOutExpected[12] = std::complex<double>(9, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(9, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(9, 2);

//    applyFilterAndCrop<float>(tmpFFTGpuIn, tmpFFTGpuOut, 1, xInFFT, yIn,
//            xOutFFT, yOut, NULL); // FIXME test

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTGpuOut[i].real();
        float cpuImag = tmpFFTGpuOut[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE GPU EO: %lu gpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingGpuOE() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
    xIn = yIn = 9;
    xOut = yOut = 6;
    xOutFFT = xOut / 2 + 1; // == 4
    xInFFT = xIn / 2 + 1; // == 5

    std::complex<float>* tmpFFTGpuIn = new std::complex<float>[yIn * xInFFT];
    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[yOut * xOutFFT];
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < yIn; ++y) {
        for (size_t x = 0; x < xInFFT; ++x) {
            size_t index = y * xInFFT + x;
            tmpFFTGpuIn[index] = std::complex<float>(y, x);
        }
    }
    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
    tmpFFTCpuOutExpected[3] = std::complex<double>(0, 3);

    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[6] = std::complex<double>(1, 2);
    tmpFFTCpuOutExpected[7] = std::complex<double>(1, 3);

    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[9] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[10] = std::complex<double>(2, 2);
    tmpFFTCpuOutExpected[11] = std::complex<double>(2, 3);

    tmpFFTCpuOutExpected[12] = std::complex<double>(3, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(3, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(3, 2);
    tmpFFTCpuOutExpected[15] = std::complex<double>(3, 3);

    tmpFFTCpuOutExpected[16] = std::complex<double>(7, 0);
    tmpFFTCpuOutExpected[17] = std::complex<double>(7, 1);
    tmpFFTCpuOutExpected[18] = std::complex<double>(7, 2);
    tmpFFTCpuOutExpected[19] = std::complex<double>(7, 3);

    tmpFFTCpuOutExpected[20] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[21] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[22] = std::complex<double>(8, 2);
    tmpFFTCpuOutExpected[23] = std::complex<double>(8, 3);

//    applyFilterAndCrop<float>(tmpFFTGpuIn, tmpFFTGpuOut, 1, xInFFT, yIn,
//            xOutFFT, yOut, NULL); // FIXME test

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTGpuOut[i].real();
        float cpuImag = tmpFFTGpuOut[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE GPU OE: %lu gpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingGpuEE() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
    xIn = yIn = 10;
    xOut = yOut = 6;
    xOutFFT = xOut / 2 + 1; // == 4
    xInFFT = xIn / 2 + 1; // == 6

    std::complex<float>* tmpFFTGpuIn = new std::complex<float>[yIn * xInFFT];
    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[yOut * xOutFFT];
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < yIn; ++y) {
        for (size_t x = 0; x < xInFFT; ++x) {
            size_t index = y * xInFFT + x;
            tmpFFTGpuIn[index] = std::complex<float>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
    tmpFFTCpuOutExpected[3] = std::complex<double>(0, 3);

    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[6] = std::complex<double>(1, 2);
    tmpFFTCpuOutExpected[7] = std::complex<double>(1, 3);

    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[9] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[10] = std::complex<double>(2, 2);
    tmpFFTCpuOutExpected[11] = std::complex<double>(2, 3);

    tmpFFTCpuOutExpected[12] = std::complex<double>(3, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(3, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(3, 2);
    tmpFFTCpuOutExpected[15] = std::complex<double>(3, 3);

    tmpFFTCpuOutExpected[16] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[17] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[18] = std::complex<double>(8, 2);
    tmpFFTCpuOutExpected[19] = std::complex<double>(8, 3);

    tmpFFTCpuOutExpected[20] = std::complex<double>(9, 0);
    tmpFFTCpuOutExpected[21] = std::complex<double>(9, 1);
    tmpFFTCpuOutExpected[22] = std::complex<double>(9, 2);
    tmpFFTCpuOutExpected[23] = std::complex<double>(9, 3);

//    applyFilterAndCrop<float>(tmpFFTGpuIn, tmpFFTGpuOut, 1, xInFFT, yIn,
//            xOutFFT, yOut, NULL); // FIXME test

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTGpuOut[i].real();
        float cpuImag = tmpFFTGpuOut[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE GPU EE: %lu gpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingCpuOO() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
    xIn = yIn = 9;
    xOut = yOut = 5;
    xOutFFT = xOut / 2 + 1; // == 3

    Image<double> inputDouble(xIn, yIn);
    Image<double> outputDouble(xOut, yOut);
    MultidimArray<std::complex<double> > tmpFFTCpuIn(yIn, xIn / 2 + 1);
    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < tmpFFTCpuIn.ydim; ++y) {
        for (size_t x = 0; x < tmpFFTCpuIn.xdim; ++x) {
            size_t index = y * tmpFFTCpuIn.xdim + x;
            tmpFFTCpuIn.data[index] = std::complex<double>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);

    tmpFFTCpuOutExpected[3] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 2);

    tmpFFTCpuOutExpected[6] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[7] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 2);

    tmpFFTCpuOutExpected[9] = std::complex<double>(7, 0);
    tmpFFTCpuOutExpected[10] = std::complex<double>(7, 1);
    tmpFFTCpuOutExpected[11] = std::complex<double>(7, 2);

    tmpFFTCpuOutExpected[12] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(8, 2);

    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
            tmpFFTCpuOut);

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTCpuOut.data[i].real();
        float cpuImag = tmpFFTCpuOut.data[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE CPU OO: %lu cpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingCpuEO() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT;
    xIn = yIn = 10;
    xOut = yOut = 5;
    xOutFFT = xOut / 2 + 1; // == 3

    Image<double> inputDouble(xIn, yIn);
    Image<double> outputDouble(xOut, yOut);
    MultidimArray<std::complex<double> > tmpFFTCpuIn(yIn, xIn / 2 + 1);
    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < tmpFFTCpuIn.ydim; ++y) {
        for (size_t x = 0; x < tmpFFTCpuIn.xdim; ++x) {
            size_t index = y * tmpFFTCpuIn.xdim + x;
            tmpFFTCpuIn.data[index] = std::complex<double>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);

    tmpFFTCpuOutExpected[3] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 2);

    tmpFFTCpuOutExpected[6] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[7] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 2);

    tmpFFTCpuOutExpected[9] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[10] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[11] = std::complex<double>(8, 2);

    tmpFFTCpuOutExpected[12] = std::complex<double>(9, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(9, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(9, 2);

    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
            tmpFFTCpuOut);

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTCpuOut.data[i].real();
        float cpuImag = tmpFFTCpuOut.data[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE CPU EO: %lu cpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingCpuOE() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT;
    xIn = yIn = 9;
    xOut = yOut = 6;
    xOutFFT = xOut / 2 + 1; // == 4

    Image<double> inputDouble(xIn, yIn);
    Image<double> outputDouble(xOut, yOut);
    MultidimArray<std::complex<double> > tmpFFTCpuIn(yIn, xIn / 2 + 1);
    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < tmpFFTCpuIn.ydim; ++y) {
        for (size_t x = 0; x < tmpFFTCpuIn.xdim; ++x) {
            size_t index = y * tmpFFTCpuIn.xdim + x;
            tmpFFTCpuIn.data[index] = std::complex<double>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
    tmpFFTCpuOutExpected[3] = std::complex<double>(0, 3);

    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[6] = std::complex<double>(1, 2);
    tmpFFTCpuOutExpected[7] = std::complex<double>(1, 3);

    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[9] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[10] = std::complex<double>(2, 2);
    tmpFFTCpuOutExpected[11] = std::complex<double>(2, 3);

    tmpFFTCpuOutExpected[12] = std::complex<double>(3, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(3, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(3, 2);
    tmpFFTCpuOutExpected[15] = std::complex<double>(3, 3);

    tmpFFTCpuOutExpected[16] = std::complex<double>(7, 0);
    tmpFFTCpuOutExpected[17] = std::complex<double>(7, 1);
    tmpFFTCpuOutExpected[18] = std::complex<double>(7, 2);
    tmpFFTCpuOutExpected[19] = std::complex<double>(7, 3);

    tmpFFTCpuOutExpected[20] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[21] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[22] = std::complex<double>(8, 2);
    tmpFFTCpuOutExpected[23] = std::complex<double>(8, 3);

    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
            tmpFFTCpuOut);

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTCpuOut.data[i].real();
        float cpuImag = tmpFFTCpuOut.data[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE CPU OE: %lu cpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingCpuEE() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT;
    xIn = yIn = 10;
    xOut = yOut = 6;
    xOutFFT = xOut / 2 + 1; // == 4

    Image<double> inputDouble(xIn, yIn);
    Image<double> outputDouble(xOut, yOut);
    MultidimArray<std::complex<double> > tmpFFTCpuIn(yIn, xIn / 2 + 1);
    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < tmpFFTCpuIn.ydim; ++y) {
        for (size_t x = 0; x < tmpFFTCpuIn.xdim; ++x) {
            size_t index = y * tmpFFTCpuIn.xdim + x;
            tmpFFTCpuIn.data[index] = std::complex<double>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
    tmpFFTCpuOutExpected[3] = std::complex<double>(0, 3);

    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[6] = std::complex<double>(1, 2);
    tmpFFTCpuOutExpected[7] = std::complex<double>(1, 3);

    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[9] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[10] = std::complex<double>(2, 2);
    tmpFFTCpuOutExpected[11] = std::complex<double>(2, 3);

    tmpFFTCpuOutExpected[12] = std::complex<double>(3, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(3, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(3, 2);
    tmpFFTCpuOutExpected[15] = std::complex<double>(3, 3);

    tmpFFTCpuOutExpected[16] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[17] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[18] = std::complex<double>(8, 2);
    tmpFFTCpuOutExpected[19] = std::complex<double>(8, 3);

    tmpFFTCpuOutExpected[20] = std::complex<double>(9, 0);
    tmpFFTCpuOutExpected[21] = std::complex<double>(9, 1);
    tmpFFTCpuOutExpected[22] = std::complex<double>(9, 2);
    tmpFFTCpuOutExpected[23] = std::complex<double>(9, 3);

    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
            tmpFFTCpuOut);

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTCpuOut.data[i].real();
        float cpuImag = tmpFFTCpuOut.data[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE CPU EE: %lu cpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testFFTAndScale() {
    double delta = 0.00001;
    size_t xIn, yIn, xOut, yOut, xOutFFT;
    xIn = yIn = 4096;
    xOut = yOut = 2276;
    xOutFFT = xOut / 2 + 1;
    size_t order = 10000;
    size_t fftPixels = xOutFFT * yOut;

    srand(42);

    Image<double> inputDouble(xIn, yIn); // keep sync with values
    Image<float> inputFloat(xIn, yIn); // keep sync with values
    size_t pixels = inputDouble.data.xdim * inputDouble.data.ydim;
    for (size_t y = 0; y < inputDouble.data.ydim; ++y) {
        for (size_t x = 0; x < inputDouble.data.xdim; ++x) {
            size_t index = y * inputDouble.data.xdim + x;
            double value = rand() / (RAND_MAX / 2000.);
            inputDouble.data.data[index] = value;
            inputFloat.data.data[index] = (float) value;
        }
    }

    float* filter = new float[fftPixels];
    for (size_t i = 0; i < fftPixels; ++i) {
        filter[i] = rand() / (float) RAND_MAX;
    }

    // CPU part
    Image<double> outputDouble(xOut, yOut);
    MultidimArray<std::complex<double> > tmpFFTCpuIn;
    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
    FourierTransformer transformer;

    transformer.FourierTransform(inputDouble(), tmpFFTCpuIn, true);
    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
            tmpFFTCpuOut);

    for (size_t nn = 0; nn < fftPixels; ++nn) {
        double wlpf = filter[nn];
        DIRECT_MULTIDIM_ELEM(tmpFFTCpuOut,nn) *= wlpf;
    }

    // store results to drive
    Image<double> fftCPU(tmpFFTCpuOut.xdim, tmpFFTCpuOut.ydim);
    for (size_t i = 0; i < fftPixels; i++) {
        fftCPU.data.data[i] = tmpFFTCpuOut.data[i].real();
    }
    fftCPU.write("testFFTCpuScaled.vol");

    // GPU part

    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[fftPixels];
    float* d_filter = loadToGPU(filter, fftPixels);

    GpuMultidimArrayAtGpu<float> gpuIn(inputFloat.data.xdim,
            inputFloat.data.ydim);
    gpuIn.copyToGpu(inputFloat.data.data);
    GpuMultidimArrayAtGpu<std::complex<float> > gpuFFT;
    mycufftHandle handle;

//    processInput(gpuIn, gpuFFT, handle, xIn, yIn, 1, xOutFFT, yOut, d_filter,
//            tmpFFTGpuOut); FIXME test

    // store results to drive
    Image<float> fftGPU(xOutFFT, yOut);
    float norm = inputFloat.data.yxdim;
    for (size_t i = 0; i < fftPixels; i++) {
        fftGPU.data.data[i] = tmpFFTGpuOut[i].real() / norm;
    }
    fftGPU.write("testFFTGpuScaled.vol");

    ////////////////////////////////////////

    if (fftCPU.data.xdim != fftGPU.data.xdim) {
        printf("wrong size: X cpu %lu X gpu %lu\n", fftCPU.data.xdim,
                fftGPU.data.xdim);
    }
    if (fftCPU.data.ydim != fftGPU.data.ydim) {
        printf("wrong size: Y cpu %lu Y gpu %lu\n", fftCPU.data.xdim,
                fftGPU.data.xdim);
    }

    for (size_t i = 0; i < fftCPU.data.yxdim; ++i) {
        float cpuReal = tmpFFTCpuOut.data[i].real();
        float cpuImag = tmpFFTCpuOut.data[i].imag();
        float gpuReal = tmpFFTGpuOut[i].real() / norm;
        float gpuImag = tmpFFTGpuOut[i].imag() / norm;
        if ((std::abs(cpuReal - gpuReal) > delta)
                || (std::abs(cpuImag - gpuImag) > delta)) {
            printf("ERROR SCALE: %lu cpu (%f, %f) gpu (%f, %f)\n", i, cpuReal,
                    cpuImag, gpuReal, gpuImag);
        }
    }
    delete[] tmpFFTGpuOut;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::loadData(const MetaData& movie,
        const Image<T>& dark, const Image<T>& gain, T targetOccupancy,
        const MultidimArray<T>& lpf) {

    setDevice(device);

    bool cropInput = (this->yDRcorner != -1);
    int noOfImgs = this->nlast - this->nfirst + 1;

    // get frame info
    Image<T> frame;
    loadFrame(movie, movie.firstObject(), cropInput, frame);
    setSizes(frame, noOfImgs);
    // prepare filter
    MultidimArray<T> filter;
    filter.initZeros(croppedOptSizeY, croppedOptSizeFFTX);
    this->scaleLPF(lpf, croppedOptSizeX, croppedOptSizeY, targetOccupancy,
            filter);

    // load all frames to RAM
    // reuse memory
    frameFourier = (std::complex<T>*)loadToRAM(movie, noOfImgs, dark, gain, cropInput);
    // scale and transform to FFT on GPU
    performFFTAndScale((T*)frameFourier, noOfImgs, inputOptSizeX,
            inputOptSizeY, inputOptBatchSize, croppedOptSizeFFTX,
            croppedOptSizeY, &filter);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::computeShifts(size_t N,
        const Matrix1D<T>& bX, const Matrix1D<T>& bY, const Matrix2D<T>& A) {
    setDevice(device);
    // since we are using different size of FFT, we need to scale results to
    // 'expected' size
    T localSizeFactorX = this->sizeFactor
            / (croppedOptSizeX / (T) inputOptSizeX);
    T localSizeFactorY = this->sizeFactor
            / (croppedOptSizeY / (T) inputOptSizeY);
    size_t scaledMaxShift = std::floor((this->maxShift * this->sizeFactor) / localSizeFactorX);

    T* correlations;
    size_t centerSize = std::ceil(scaledMaxShift * 2 + 1);
    computeCorrelations(centerSize, N, frameFourier, croppedOptSizeFFTX,
            croppedOptSizeX, croppedOptSizeY, correlationBufferImgs,
            croppedOptBatchSize, correlations);


    int idx = 0;
    MultidimArray<T> Mcorr(centerSize, centerSize);
    for (size_t i = 0; i < N - 1; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            size_t offset = idx * centerSize * centerSize;
            Mcorr.data = correlations + offset;
            Mcorr.setXmippOrigin();
            bestShift(Mcorr, bX(idx), bY(idx), NULL, scaledMaxShift);
            bX(idx) *= localSizeFactorX; // scale to expected size
            bY(idx) *= localSizeFactorY;
            if (this->verbose)
                std::cerr << "Frame " << i + this->nfirst << " to Frame "
                        << j + this->nfirst << " -> ("
                        << bX(idx) / this->sizeFactor << ","
                        << bY(idx) / this->sizeFactor << ")" << std::endl;
            for (int ij = i; ij < j; ij++)
                A(idx, ij) = 1;

            idx++;
        }
    }
    Mcorr.data = NULL;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::releaseGlobalAlignResources() {
	// GPU memory has been already freed
	delete[] frameFourier;
}

// explicit specialization
template class ProgMovieAlignmentCorrelationGPU<float> ;
