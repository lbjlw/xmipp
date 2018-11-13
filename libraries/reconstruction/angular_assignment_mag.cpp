/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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

#include "angular_assignment_mag.h"

void ProgAngularAssignmentMag::defineParams()
{
    //usage
    addUsageLine("Generate 3D reconstructions from projections using random orientations");
    //params
    addParamsLine("   -i <md_file>               : Metadata file with input experimental projections");
    addParamsLine("   -ref <md_file>             : Metadata file with input reference projections");
    addParamsLine("  [-odir <outputDir=\".\">]   : Output directory");
    addParamsLine("  [-sym <symfile=c1>]         : Enforce symmetry in projections");
}

// Read arguments ==========================================================
void ProgAngularAssignmentMag::readParams()
{
    fnIn = getParam("-i");
    fnRef = getParam("-ref");
    fnDir = getParam("-odir");
    fnSym = getParam("-sym");
}

// Show ====================================================================
void ProgAngularAssignmentMag::show()
{
    if (verbose > 0)
    {
        std::cout << "Input metadata              : "  << fnIn        << std::endl;
        std::cout << "Input references            : "  << fnRef       << std::endl;
        std::cout << "Output directory            : "  << fnDir       << std::endl;
        if (fnSym != "")
            std::cout << "Symmetry for projections    : "  << fnSym << std::endl;
    }
}

// Main routine ------------------------------------------------------------
void ProgAngularAssignmentMag::run()
{
    produceSideinfo(); // read metaData file

    FileName fnImgExp;
    FileName fnImgRef;
    MDRow rowExp, rowRef;
    int countInImg = 0, countRefImg = 0;
    // reading input stack image
    MDIterator *iterExp = new MDIterator(mdIn);
    int sizeMdIn = mdIn.size();
    size_t Zdim, Ndim;
    getImageSize(mdIn,Xdim,Ydim,Zdim,Ndim);
    /*std::cerr << "Size Exp: " << sizeMdIn <<
                 " Xdim_in: "<< Xdim <<
                 " Ydim_in: "<< Ydim <<
                 " Ndim_in: "<< Ndim <<
                 " Zdim_in: "<< Zdim << std::endl;*/

    // reading reference stack image
    MDIterator *iterRef = new MDIterator(mdRef);
    int sizeMdRef = mdRef.size();
    size_t XdimRef, YdimRef, ZdimRef, NdimRef;
    getImageSize(mdRef,XdimRef, YdimRef, ZdimRef, NdimRef);
    /*std::cerr << "Size Ref: " << sizeMdRef <<
                 " Xdim_ref: "<< XdimRef <<
                 " Ydim_ref: "<< YdimRef <<
                 " Ndim_ref: "<< NdimRef <<
                 " Zdim_ref: "<< ZdimRef << std::endl;*/

    // passing images to Image and then to MultidimArray structure
    const size_t n_bands = 16;
    const size_t startBand = 5;
    const size_t n_rad = size_t(Xdim/2 + 0.5);
    const size_t n_ang = size_t(360);

    // init "delay axes"
    _delayAxes(Ydim, Xdim, n_ang);


    // experimental image related
    Image<double>                           ImgIn;
    MultidimArray<double>                   MDaIn(Ydim,Xdim);
    MultidimArray< std::complex<double> >   MDaInF(Ydim, Xdim);
    MultidimArray<double>                   MDaInFM(Ydim, Xdim);
    MultidimArray<double>                   MDaInFMs(Ydim, Xdim);
    MultidimArray<double>                   MDaInFMs_polar(n_rad,n_ang);
    MultidimArray<double>                   MDaInFMs_polarPart(n_bands, n_ang);
    MultidimArray< std::complex<double> >   MDaInFMs_polarF(n_bands, n_ang);

    // reference image related
    Image<double>                           ImgRef;
    MultidimArray<double>                   MDaRef(Ydim,Xdim);
    MultidimArray< std::complex<double> >   MDaRefF(Ydim, Xdim);
    MultidimArray<double>                   MDaRefFM(Ydim, Xdim);
    MultidimArray<double>                   MDaRefFMs(Ydim, Xdim);
    MultidimArray<double>                   MDaRefFMs_polar(n_rad,n_ang);
    MultidimArray<double>                   MDaRefFMs_polarPart(n_bands, n_ang);
    MultidimArray< std::complex<double> >   MDaRefFMs_polarF(n_bands, n_ang);

    // CCV result matrix
    MultidimArray<double>                   ccMatrixRot(n_bands, n_ang);
    MultidimArray<double>                   ccVectorRot( (const size_t) 1, n_ang);
    std::vector<double>                     cand; // rotation candidates
    int                                     peaksFound = 0; // peaksFound in ccVectorRot
    double                                  tempCoeff;

    // candidates for each loop
    std::vector<unsigned int>               candidatesFirstLoop(sizeMdRef,0);
    std::vector<unsigned int>               Idx(sizeMdRef,0);
    std::vector<double>                     candidatesFirstLoopCoeff(sizeMdRef,0);
    std::vector<double>                     bestTx(sizeMdRef,0);
    std::vector<double>                     bestTy(sizeMdRef,0);
    std::vector<double>                     bestRot(sizeMdRef,0);

    std::clock_t inicio, fin;
    inicio = std::clock();
    std::ofstream outfile("/home/jeison/Escritorio/outfile.txt");
    // main loop, input stack
    double psiVal, realTx, realTy;
    for (countInImg = 24500; countInImg < 24501/*sizeMdIn*/; countInImg += 30  /*countInImg++*/){
        // read experimental image
        mdIn.getRow(rowExp, size_t(countInImg+1) /*iterExp->objId*/);
        rowExp.getValue(MDL_IMAGE, fnImgExp);
        // get real values
        rowExp.getValue(MDL_ANGLE_PSI,psiVal);
        rowExp.getValue(MDL_SHIFT_X,realTx);
        rowExp.getValue(MDL_SHIFT_Y,realTy);
        //        printf("\n** %d **\n", countInImg+1 /*iterRef->objId*/);
        outfile << "\n**" << countInImg+1 <<"**\n";
        //                std::cout << "Inp image: " << fnImgExp << std::endl;
        // processing input image
        ImgIn.read(fnImgExp);
        MDaIn = ImgIn(); // getting image
        _applyFourier(MDaIn, MDaInF); // fourier experimental image (genera copia?)
        _getComplexMagnitude(MDaInF, MDaInFM); // magnitude espectra experimental image
        completeFourierShift(MDaInFM, MDaInFMs); // shift spectrum
        MDaInFMs_polar = imToPolar(MDaInFMs, n_rad, n_ang); // polar representation of magnitude
        selectBands(MDaInFMs_polar, MDaInFMs_polarPart, n_bands, startBand, n_ang); // select bands
        _applyFourier(MDaInFMs_polarPart, MDaInFMs_polarF); // apply fourier

        // "restart" iterator for reference image
        iterRef->init(mdRef);
        tempCoeff = -10.0;
        int k = 0;
        double bestCandVar, bestCoeff, Tx, Ty;

        // loop over reference stack
        for(countRefImg = 0; countRefImg < sizeMdRef; countRefImg++){
            mdRef.getRow(rowRef, size_t(countRefImg+1) /*iterRef->objId*/);
            rowRef.getValue(MDL_IMAGE, fnImgRef);

            // processing reference image
            ImgRef.read(fnImgRef);
            MDaRef = ImgRef();
            _applyFourier(MDaRef, MDaRefF);// fourier experimental image (genera copia?)
            _getComplexMagnitude(MDaRefF, MDaRefFM);// magnitude espectra experimental image
            completeFourierShift(MDaRefFM, MDaRefFMs);// shift spectrum
            MDaRefFMs_polar = imToPolar(MDaRefFMs, n_rad, n_ang);// polar representation of magnitude
            selectBands(MDaRefFMs_polar, MDaRefFMs_polarPart, n_bands, startBand, n_ang); // select bands
            _applyFourier(MDaRefFMs_polarPart,MDaRefFMs_polarF); // apply fourier

            // computing relative rotation and traslation
            ccMatrix(MDaInFMs_polarF, MDaRefFMs_polarF, ccMatrixRot);// cross-correlation matrix
            maxByColumn(ccMatrixRot, ccVectorRot, n_bands, n_ang); // ccvMatrix to ccVector
            peaksFound = 0;
            std::vector<double>().swap(cand); // alternative to cand.clear(), which wasn't working
            rotCandidates(ccVectorRot, cand, n_ang, &peaksFound); // compute condidates set {\theta + 180}

            // bestCand method return best cand rotation and its correspondient tx, ty and coeff
            bestCand(MDaIn, MDaInF, MDaRef, cand, peaksFound, &bestCandVar, &Tx, &Ty, &bestCoeff);

            // all the results are storaged for posterior partial_sort
            Idx[countRefImg] = k++;
            candidatesFirstLoop[countRefImg] = countRefImg+1;
            candidatesFirstLoopCoeff[countRefImg] = bestCoeff;
            bestTx[countRefImg] = Tx;
            bestTy[countRefImg] = Ty;
            bestRot[countRefImg] = bestCandVar;

            // next reference
            if(iterRef->hasNext())
                iterRef->moveNext();

        }
        // choose nCand of the candidates with best corrCoeff
        int nCand = 25;
        std::partial_sort(Idx.begin(), Idx.begin()+nCand, Idx.end(),
                          [&](int i, int j){return candidatesFirstLoopCoeff[i] > candidatesFirstLoopCoeff[j]; });

        // second loop applies search only over better candidates
        k = 0;
        // candidates second loop
        std::vector<unsigned int>               candidatesFirstLoop2(nCand,0);
        std::vector<unsigned int>               Idx2(nCand,0);
        std::vector<double>                     candidatesFirstLoopCoeff2(nCand,0);
        std::vector<double>                     bestTx2(nCand,0);
        std::vector<double>                     bestTy2(nCand,0);
        std::vector<double>                     bestRot2(nCand,0);
        MultidimArray<double>                   MDaRefTrans;

        for (int i = 0; i < nCand; i++){
            mdRef.getRow(rowRef, size_t(candidatesFirstLoop[ Idx[i] ]));
            rowRef.getValue(MDL_IMAGE, fnImgRef);
            // En lugar de hacer todo esto podria almacenar los candidatos a rotación del primer loop
            // processing reference image
            ImgRef.read(fnImgRef);
            MDaRef = ImgRef();
            // aplicar dicha rotación a la imagen referencia y volver a calcular rotación y traslación
            double rotVal = bestRot[ Idx[i] ];
            double trasXval = bestTx[ Idx[i] ];
            double trasYval = bestTy[ Idx[i] ];
            _applyRotationAndShift(MDaRef, rotVal, trasXval, trasYval, MDaRefTrans);
            _applyFourier(MDaRefTrans, MDaRefF);// fourier experimental image (genera copia?)
            _getComplexMagnitude(MDaRefF, MDaRefFM);// magnitude espectra experimental image
            completeFourierShift(MDaRefFM, MDaRefFMs);// shift spectrum
            MDaRefFMs_polar = imToPolar(MDaRefFMs, n_rad, n_ang);// polar representation of magnitude
            selectBands(MDaRefFMs_polar, MDaRefFMs_polarPart, n_bands, startBand, n_ang); // select bands
            _applyFourier(MDaRefFMs_polarPart,MDaRefFMs_polarF); // apply fourier

            // computing relative rotation and traslation
            ccMatrix(MDaInFMs_polarF, MDaRefFMs_polarF, ccMatrixRot);// cross-correlation matrix
            maxByColumn(ccMatrixRot, ccVectorRot, n_bands, n_ang); // ccvMatrix to ccVector
            peaksFound = 0;
            std::vector<double>().swap(cand); // alternative to cand.clear(), which wasn't working
            rotCandidates(ccVectorRot, cand, n_ang, &peaksFound); // compute condidates set {\theta + 180}

            // bestCand method return best cand rotation and its correspondient tx, ty and coeff
            bestCand2(MDaIn, MDaInF, MDaRefTrans, cand, peaksFound, &bestCandVar, &Tx, &Ty, &bestCoeff);

            // todos los datos son almacenados para partial_sort posterior (ahora)
            Idx2[i] = k++;
            candidatesFirstLoop2[i] = candidatesFirstLoop[ Idx[i] ];
            candidatesFirstLoopCoeff2[i] = bestCoeff;
            bestTx2[i] = Tx + trasXval;
            bestTy2[i] = Ty + trasYval;
            bestRot2[i] = bestCandVar + rotVal;

        }
        // mostrar segundo ordenamiento
        // choose nCand of the candidates with best corrCoeff
        int nCand2 = 5;
        std::partial_sort(Idx2.begin(), Idx2.begin()+nCand2, Idx2.end(),
                          [&](int i, int j){return candidatesFirstLoopCoeff2[i] > candidatesFirstLoopCoeff2[j]; });

        //std::cout << "input image " << countInImg + 1 << ", direction candidates: \n";
        outfile << "size of Idx2Vector underwent to second sorting: " << Idx2.size() << "\n";
        outfile << "input image " << countInImg + 1 << ", direction candidates: \n";

        for(int i = 0; i < nCand2; i++)
            outfile   << "dir:  "          << candidatesFirstLoop2[ Idx2[i] ]
                      << "\t    coef:  "    << candidatesFirstLoopCoeff2[Idx2[i]]
                      << "\t    rot:  "     << bestRot2[Idx2[i]]
                      << "\t    tx:  "      << bestTx2[Idx2[i]]
                      << "\t    ty:  "      << bestTy2[Idx2[i]] << "\n";
        outfile << "\t \t real Rot/psi: "   << psiVal
                << "\t    realTx: "         << realTx
                << "\t    realTy: "         << realTy << "\n";
        outfile << "\n";

        // next experimental
        if(iterExp->hasNext())
            iterExp->moveNext();
    }
    fin = std::clock();

    delete iterExp;
    delete iterRef;
    transformer.cleanup();

    //std::cout << "elapsed time (min): "    << (double)((fin - inicio)/CLOCKS_PER_SEC)/60. << std::endl;
    double eTime = (double)((fin - inicio)/CLOCKS_PER_SEC);
    outfile << "elapsed time (s): "    << eTime << "\n";
    outfile << "elapsed time (min): "  << eTime/60. << "\n";
    outfile.close();
}

/* print in console some values of double MultidimArray */
void ProgAngularAssignmentMag::printSomeValues(MultidimArray<double> &MDa){
    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            std::cout << "val: " << DIRECT_A2D_ELEM(MDa,i,j) << std::endl;
}

void ProgAngularAssignmentMag::produceSideinfo()
{
    mdIn.read(fnIn);
    mdRef.read(fnRef);

}

/* Pearson Coeff */
void ProgAngularAssignmentMag::pearsonCorr(MultidimArray<double> &X, MultidimArray<double> &Y, double &coeff){

    //     //no good results
    //    MultidimArray<int>      mask(Ydim,Xdim);
    //    MultidimArray<double>   X2(Ydim,Xdim);
    //    MultidimArray<double>   Y2(Ydim,Xdim);
    //    mask.setXmippOrigin();
    //    BinaryCircularMask(mask,Xdim/2);
    //    apply_binary_mask(mask, X, X2);
    //    apply_binary_mask(mask, Y, Y2);
    // covariance
    double X_m, Y_m, X_std, Y_std;
    arithmetic_mean_and_stddev(X, X_m, X_std);
    arithmetic_mean_and_stddev(Y, Y_m, Y_std);
    //    std::cout << "X_m, Y_m, X_std, Y_std: " << X_m <<", "<< Y_m <<", "<< X_std <<", "<< Y_std << std::endl;

    double prod_mean = mean_of_products(X, Y);
    double covariace = prod_mean - (X_m * Y_m);

    coeff = covariace / (X_std * Y_std);
}

/* Arithmetic mean and stdDev for Pearson Coeff */
void ProgAngularAssignmentMag::arithmetic_mean_and_stddev( MultidimArray<double> &data, double &avg, double &stddev ){
    data.computeAvgStdev(avg, stddev);
}

/* Mean of products for Pearson Coeff */
double ProgAngularAssignmentMag::mean_of_products(MultidimArray<double> &data1, MultidimArray<double> &data2){
    double total = 0;
    for (int f = 0; f < Ydim; f++){
        for (int c = 0; c < Xdim; c++){
            total += DIRECT_A2D_ELEM(data1,f,c) * DIRECT_A2D_ELEM(data2,f,c);
        }
    }
    return total/(Xdim*Ydim);
}

/* writing out some data to file with an specified size*/
void ProgAngularAssignmentMag::_writeTestFile(MultidimArray<double> &data, const char* fileName,
                                              size_t nFil, size_t nCol){
    std::ofstream outFile(fileName);
    for (int f = 0; f < nFil; f++){
        for (int c = 0; c < nCol; c++){
            outFile <<  DIRECT_A2D_ELEM(data,f,c) << "\t";
        }
        outFile << "\n";
    }
    outFile.close();
}

/* writing out some data to file Ydim x Xdim size*/
void ProgAngularAssignmentMag::_writeTestFile(MultidimArray<double> &data, const char* fileName){
    std::ofstream outFile(fileName);
    for (int f = 0; f < Ydim; f++){
        for (int c = 0; c < Xdim; c++){
            outFile <<  DIRECT_A2D_ELEM(data,f,c) << "\t";
        }
        outFile << "\n";
    }
    outFile.close();
}

/* get COMPLETE fourier spectrum. It should be changed for half */
void ProgAngularAssignmentMag::_applyFourier(MultidimArray<double> &data,
                                             MultidimArray< std::complex<double> > &FourierData){


    // esta opción retorna una copia completa (espero más adelante usar solo la mitad y no hacer copia)
    transformer.completeFourierTransform(data, FourierData);

    // transformer.FourierTransform(data, FourierData, false);
    // transformer.cleanup();

    /* Example How to use, from xmipp_fftw
     * FourierTransformer transformer;
     * MultidimArray< std::complex<double> > Vfft;
     * transformer.FourierTransform(V(),Vfft,false);
     * MultidimArray<double> Vmag;
     * Vmag.resize(Vfft);
     * FOR_ALL_ELEMENTS_IN_ARRAY3D(Vmag)
     *     Vmag(k,i,j)=20*log10(abs(Vfft(k,i,j)));
    */
}

/* get magnitude of fourier spectrum */
void ProgAngularAssignmentMag::_getComplexMagnitude( MultidimArray< std::complex<double> > &FourierData,
                                                     MultidimArray<double> &FourierMag){
    FFT_magnitude(FourierData,FourierMag);
}

/* cartImg contains cartessian  grid representation of image,
*  rad and ang are the number of radius and angular elements*/
MultidimArray<double> ProgAngularAssignmentMag::imToPolar(MultidimArray<double> &cartIm,
                                                          const size_t &rad, const size_t &ang){
    MultidimArray<double> polarImg(rad, ang);
    float pi = 3.141592653;
    // coordinates of center
    double cy = (Ydim+1)/2.0;
    double cx = (Xdim+1)/2.0;
    // scale factors
    double sfy = (Ydim-1)/2.0;
    double sfx = (Xdim-1)/2.0;

    double delR = (double)(1.0 / (rad-1));
    double delT = 2.0 * pi / ang;

    // loop through rad and ang coordinates
    double r, t, x_coord, y_coord;
    for(size_t ri = 0; ri < rad; ri++){
        for(size_t ti = 0; ti < ang; ti++ ){
            r = ri * delR;
            t = ti * delT;
            x_coord = ( r * cos(t) ) * sfx + cx;
            y_coord = ( r * sin(t) ) * sfy + cy;
            // set value of polar img
            DIRECT_A2D_ELEM(polarImg,ri,ti) = interpolate(cartIm,x_coord,y_coord);
        }
    }

    return polarImg;
}

/* bilinear interpolation */
double ProgAngularAssignmentMag::interpolate(MultidimArray<double> &cartIm,
                                             double &x_coord, double &y_coord){
    double val;
    size_t xf = floor(x_coord);
    size_t xc = ceil(x_coord);
    size_t yf = floor(y_coord);
    size_t yc = ceil(y_coord);

    if ( (xf == xc) && ( yf == yc )){
        val = dAij(cartIm, xc, yc);
    }
    else if (xf == xc){ // linear
        val = dAij(cartIm, xf, yf) + (y_coord - yf) * ( dAij(cartIm, xf, yc) - dAij(cartIm, xf, yf) );
    }
    else if(yf == yc){ // linear
        val = dAij(cartIm, xf, yf) + (x_coord - xf) * ( dAij(cartIm, xc, yf) - dAij(cartIm, xf, yf) );
    }
    else{ // bilinear
        val = ((double)(( dAij(cartIm,xf,yf)*(yc-y_coord) + dAij(cartIm,xf,yc)*(y_coord-yf) ) * (xc - x_coord)) +
               (double)(( dAij(cartIm,xc,yf)*(yc-y_coord) + dAij(cartIm,xc,yc)*(y_coord-yf) ) * (x_coord - xf))
               )  / (double)( (xc - xf)*(yc - yf) );
    }

    return val;

}

/* its an experiment for implement fftshift*/
void ProgAngularAssignmentMag::completeFourierShift(MultidimArray<double> &in, MultidimArray<double> &out){
    size_t Cf = (size_t)(Ydim/2.0 + 0.5);
    size_t Cc = (size_t)(Xdim/2.0 + 0.5);

    size_t ff, cc;
    for(size_t f = 0; f < Ydim; f++){
        ff = (f + Cf) % Ydim;
        for(size_t c = 0; c < Xdim; c++){
            cc = (c + Cc) % Xdim;
            DIRECT_A2D_ELEM(out, ff, cc) = DIRECT_A2D_ELEM(in,f,c);
        }
    }
}

/* experiment for GCC matrix product F1 .* conj(F2)
* here F1 is a copy
*/
void ProgAngularAssignmentMag::ccMatrix(MultidimArray< std::complex<double>> F1,
                                        MultidimArray< std::complex<double>> &F2,
                                        MultidimArray<double> &result){
    // Multiply F1 * F2' -- from fast_correlation_vector in xmipp_fftw
    double a, b, c, d; // a+bi, c+di
    double *ptrFFT2=(double*)MULTIDIM_ARRAY(F2);
    double *ptrFFT1=(double*)MULTIDIM_ARRAY(F1);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F1){
        a=*ptrFFT1;
        b=*(ptrFFT1+1);
        c=(*ptrFFT2++);
        d=(*ptrFFT2++)*(-1);
        *ptrFFT1++ = a*c-b*d;
        *ptrFFT1++ = b*c+a*d;
    }

    // Invert the product, in order to obtain the correlation image
    InverseFourierTransform(F1,result);

    // Center the resulting image to obtain a centered autocorrelation
    CenterFFT(result, true);
    result.setXmippOrigin();
    //    _writeTestFile(result, "/home/jeison/Escritorio/t_result.txt", n_bands, n_angs);
}

/* select n_bands of polar representation of magnitude spectrum */
void ProgAngularAssignmentMag::selectBands(MultidimArray<double> &in, MultidimArray<double> &out,
                                           const size_t &n_bands, const size_t &startBand, const size_t &n_ang ){
    for (int i = 0; i < n_bands; i++){
        for (int j = 0; j < n_ang; j++){
            dAij(out,i,j) = dAij(in, startBand+i, j);
        }
    }
}

/* gets maximum value for each column*/
void ProgAngularAssignmentMag::maxByColumn(MultidimArray<double> &in,
                                           MultidimArray<double> &out,
                                           const size_t &nFil, const size_t &nCol){
    int f, c;
    double maxVal, val2;
    for(c = 0; c < nCol; c++){
        maxVal = dAij(in, 0, c);
        for(f = 1; f < nFil; f++){
            val2 = dAij(in, f, c);
            if (val2 > maxVal)
                maxVal = val2;
        }
        dAi(out,c) = maxVal;
    }
}

/* gets maximum value for each row */
void ProgAngularAssignmentMag::maxByRow(MultidimArray<double> &in,
                                        MultidimArray<double> &out,
                                        const size_t &nFil, const size_t &nCol){
    int f, c;
    double maxVal, val2;
    for(f = 0; f < nFil; f++){
        maxVal = dAij(in, f, 0);
        for(c = 1; c < nCol; c++){
            val2 = dAij(in, f, c);
            if (val2 > maxVal)
                maxVal = val2;
        }
        dAi(out,f) = maxVal;
    }
}

/* candidates to best rotation*/
void ProgAngularAssignmentMag::rotCandidates(MultidimArray<double> &in,
                                             std::vector<double> &cand,
                                             const size_t &size, int *nPeaksFound){
    const int maxNumPeaks = 100; // revisar este número ??
    int maxAccepted = 4;
    int *peakPos = (int*) calloc(maxNumPeaks,sizeof(int));
    int cont = 0;
    *(nPeaksFound) = cont;
    //    printf("nPeaksFound bef = %d\n", *(nPeaksFound));
    int i;
    for(i = 89/*1*/; i < 271/*size-1*/; i++){ // only look for in range -90:90
        if( *(nPeaksFound) > maxNumPeaks){
            printf("reaches max number of peaks!\n");
            i = size;
        }
        if ( (dAi(in,i) > dAi(in,i-1)) && (dAi(in,i) > dAi(in,i+1)) ){ // what about equal values? find peaks of soft curves
            peakPos[cont] = i;
            cont++;
            *(nPeaksFound) = cont;
        }
    }
    //    printf("nPeaksFound after = %d\n", *(nPeaksFound));
    maxAccepted = ( *(nPeaksFound) < maxAccepted) ? *(nPeaksFound) : maxAccepted;

    if(cont){
        std::vector<int> temp(*(nPeaksFound),0);
        for(i = 0; i < *(nPeaksFound); i++){
            temp[i] = peakPos[i];
            //                        printf("temp(%d)= %d\n", i, temp[i]);
        }
        // delete peakPos
        free(peakPos);

        // sorting first in case there are more than maxAccepted peaks
        std::sort(temp.begin(), temp.end(), [&](int i, int j){return dAi(in,i) > dAi(in,j); } );

        //                for(i = 0; i < *(nPeaksFound); i++){
        //                    std::cout << "Ordered temp("<<i<<")= "<<temp[i]<<"\t value= "<<dAi(in,temp[i])<<std::endl;
        ////                    printf("Ordered_temp(%d)= %d \t value = %.3f\n", i, temp[i], dAi(in, temp[i]));
        //                }

        int tam = 2*maxAccepted;
        *(nPeaksFound) = tam;
        cand.reserve(tam);
        //    std::vector<double> out(tam,0);
        for(i = 0; i < maxAccepted; i++){
            cand[i] = dAi(axRot,temp[i]);//(dAi(axRot,temp[i])>0) ? dAi(axRot,temp[i]) + 1. : dAi(axRot,temp[i]) - 1.;
            cand[i+maxAccepted] =(cand[i]>0) ? cand[i] + 180 : cand[i] - 180 ; // +-181
        }
    }
    else{
        printf("no peaks found!\n");
    }

}

/* instace of "delay axes" for assign rotation and traslation candidates*/
void ProgAngularAssignmentMag::_delayAxes(const size_t &Ydim, const size_t &Xdim, const size_t &n_ang){
    axRot.resize(1,1,1,n_ang);
    axTx.resize(1,1,1,Xdim);
    axTy.resize(1,1,1,Ydim);

    double M = double(n_ang - 1)/2.;
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(axRot){
        dAi(axRot,i) = M - i;
    }
    M = double(Xdim - 1)/2.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(axTx){
        dAi(axTx,i) = M - i;
    }
    M = double(Ydim - 1)/2.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(axTy){
        dAi(axTy,i) = M - i;
    }
}

/* selection of best candidate to rotation and its corresponding shift
 * called at first loop in "coarse" searching
 * shitfs are computed as maximum of CrossCorr vector
 * vector<double> cand contains candidates to relative rotation between images
*/
void ProgAngularAssignmentMag::bestCand(/*inputs*/
                                        MultidimArray<double> &MDaIn,
                                        MultidimArray< std::complex<double> > &MDaInF,
                                        MultidimArray<double> &MDaRef,
                                        std::vector<double> &cand,
                                        int &peaksFound,
                                        /*outputs*/
                                        double *bestCandRot,
                                        double *shift_x,
                                        double *shift_y,
                                        double *bestCoeff){
    *(bestCandRot) = 0;
    *(shift_x) = 0.;
    *(shift_y) = 0.;
    *(bestCoeff) = -10.0;
    double rotVar = 0.0;
    double tempCoeff;
    double tx, ty;
    //std::vector<double> vTx, vTy;
    MultidimArray<double> MDaRefRot;
    MultidimArray<double> MDaRefRotShift;
    MultidimArray<double> ccMatrixShift(Ydim,Xdim);
    MultidimArray<double> ccVectorTx( (const size_t) 1,Xdim);
    MultidimArray<double> ccVectorTy( (const size_t) 1,Ydim);
    MultidimArray< std::complex<double> > MDaRefRotF(Ydim, Xdim);

    MDaRefRot.setXmippOrigin();
    for(int i = 0; i < peaksFound; i++){
        rotVar = -1. * cand[i];
        _applyRotation(MDaRef,rotVar,MDaRefRot); // rotate reference images
        //std::cout << "rotacion: " << rotVar << std::endl;
        _applyFourier(MDaRefRot,MDaRefRotF); // fourier --> F2_r
        // computing relative traslation of rotated reference
        ccMatrix(MDaInF, MDaRefRotF, ccMatrixShift);// cross-correlation matrix
        maxByColumn(ccMatrixShift, ccVectorTx, Ydim, Xdim); // ccvMatrix to ccVector
        getShift(axTx, ccVectorTx,tx,Xdim);
        tx = -1. * tx;
        maxByRow(ccMatrixShift, ccVectorTy, Ydim, Xdim); // ccvMatrix to ccVector
        getShift(axTy, ccVectorTy,ty,Ydim);
        ty = -1. * ty;

        //        _writeTestFile(ccVectorTx,"/home/jeison/Escritorio/t_ccTxVector.txt", 1, Xdim);
        //        _writeTestFile(ccVectorTy,"/home/jeison/Escritorio/t_ccTyVector.txt", 1, Ydim);
        //        std::cout << "tx= " <<tx << std::endl;
        //        std::cout << "ty= " <<ty << std::endl;
        //        std::cin.ignore();
        // translate rotated version of MDaRef
        _applyShift(MDaRefRot, tx, ty, MDaRefRotShift);
        /*_writeTestFile(MDaRef,"/home/jeison/Escritorio/t_F.txt");
        _writeTestFile(MDaRefRotShift,"/home/jeison/Escritorio/t_R.txt");
        std::cout << rotVar << "\t" << tx << "\t" << ty << std::endl;
        std::cin.ignore();*/
        // Pearson coeff
        pearsonCorr(MDaIn, MDaRefRotShift, tempCoeff);
        //        // SSIM index
        //        ssimIndex(MDaIn, MDaRefRotShift, tempCoeff);
        //        std::cout << "myCorr(f1,f2_rt): " << tempCoef << std::endl;
        if ( tempCoeff > *(bestCoeff) ){
            *(bestCoeff) = tempCoeff;
            *(shift_x) = tx;
            *(shift_y) = ty;
            *(bestCandRot) = rotVar;
        }

    }
    //    std::cout << "ang1 = " << *(bestCandRot) << std::endl;
    //    // set rank [-180, 180]
    //    if ( std::abs( *(bestCandRot) ) > 180. )
    //        *(bestCandRot) = -1. * copysign(1., *(bestCandRot)) * ( 360. - std::abs( *(bestCandRot) ) );
    //    std::cout << "ang2 = " << *(bestCandRot) <<
    //                 "\t Tx = " << *(shift_x) <<
    //                 "\t Ty = " << *(shift_y) <<
    //                 "\t maxCoeff = " << *(bestCoeff) << std::endl;
}

/* apply rotation */
void ProgAngularAssignmentMag::_applyRotation(MultidimArray<double> &MDaRef, double &rot,
                                              MultidimArray<double> &MDaRefRot){
    // Transform matrix
    Matrix2D<double> A(3,3);
    A.initIdentity();
    double ang, cosine, sine;
    ang = DEG2RAD(rot);
    cosine = cos(ang);
    sine = sin(ang);

    // rotation
    MAT_ELEM(A,0, 0) = cosine;
    MAT_ELEM(A,0, 1) = sine;
    MAT_ELEM(A,1, 0) = -sine;
    MAT_ELEM(A,1, 1) = cosine;

    // Shift
    MAT_ELEM(A,0, 2) = 0.;
    MAT_ELEM(A,1, 2) = 0.;

    //    applyGeometry(LINEAR, Mref, proj_ref[refno], A, IS_NOT_INV, DONT_WRAP);
    applyGeometry(LINEAR, MDaRefRot, MDaRef, A, IS_NOT_INV, DONT_WRAP);

}

/* apply traslation */
void ProgAngularAssignmentMag::_applyShift(MultidimArray<double> &MDaRef,
                                           double &tx, double &ty,
                                           MultidimArray<double> &MDaRefShift){
    // Transform matrix
    Matrix2D<double> A(3,3);
    A.initIdentity();

    // Shift
    MAT_ELEM(A,0, 2) = tx;
    MAT_ELEM(A,1, 2) = ty;

    applyGeometry(LINEAR, MDaRefShift, MDaRef, A, IS_NOT_INV, DONT_WRAP);
}

/* finds shift as maximum of ccVector */
void ProgAngularAssignmentMag::getShift(MultidimArray<double> &axis,
                                        MultidimArray<double> &ccVector, double &shift, const size_t &size){
    double maxVal = -10.;
    int idx;
    int i;
    for(i = 0; i < size; i++){
        if(ccVector[i] > maxVal){
            maxVal = ccVector[i];
            idx = i;
        }
    }
    shift = dAi(axis, idx);
}


/* Structural similarity SSIM index Coeff */
void ProgAngularAssignmentMag::ssimIndex(MultidimArray<double> &X, MultidimArray<double> &Y, double &coeff){

    // covariance
    double X_m, Y_m, X_std, Y_std;
    double c1, c2, L;
    arithmetic_mean_and_stddev(X, X_m, X_std);
    arithmetic_mean_and_stddev(Y, Y_m, Y_std);
    //    std::cout << "X_m, Y_m, X_std, Y_std: " << X_m <<", "<< Y_m <<", "<< X_std <<", "<< Y_std << std::endl;

    double prod_mean = mean_of_products(X, Y);
    double covariace = prod_mean - (X_m * Y_m);

    L = 1; //std::pow(2.0,16) - 1.; // debe haber otra forma de obtener es mismo valor sin usar operador pow
    c1 = (0.01*L) * (0.01*L);
    c2 = (0.03*L) * (0.03*L); // estabilidad en división


    coeff = ( (2*X_m*Y_m + c1)*(2*covariace+c2) )/( (X_m*X_m + Y_m*Y_m + c1)*(X_std*X_std + Y_std*Y_std + c2) );
}

/* selection of best candidate to rotation and its corresponding shift
 * called at second loop in a little bit more strict searching
 * shitfs are computed as maximum of CrossCorr vector +0.5 / -0.5
 * vector<double> cand contains candidates to relative rotation between images with larger CrossCorr-coeff after first loop
*/
void ProgAngularAssignmentMag::bestCand2(/*inputs*/
                                        MultidimArray<double> &MDaIn,
                                        MultidimArray< std::complex<double> > &MDaInF,
                                        MultidimArray<double> &MDaRef,
                                        std::vector<double> &cand,
                                        int &peaksFound,
                                        /*outputs*/
                                        double *bestCandRot,
                                        double *shift_x,
                                        double *shift_y,
                                        double *bestCoeff){
    *(bestCandRot) = 0;
    *(shift_x) = 0.;
    *(shift_y) = 0.;
    *(bestCoeff) = -10.0;
    double rotVar = 0.0;
    double tempCoeff;
    double tx, ty;
    std::vector<double> vTx, vTy;
    MultidimArray<double> MDaRefRot;
    MultidimArray<double> MDaRefRotShift;
    MultidimArray<double> ccMatrixShift(Ydim,Xdim);
    MultidimArray<double> ccVectorTx( (const size_t) 1,Xdim);
    MultidimArray<double> ccVectorTy( (const size_t) 1,Ydim);
    MultidimArray< std::complex<double> > MDaRefRotF(Ydim, Xdim);

    MDaRefRot.setXmippOrigin();
    for(int i = 0; i < peaksFound; i++){
        rotVar = -1. * cand[i];
        _applyRotation(MDaRef,rotVar,MDaRefRot); // rotate reference images
        //std::cout << "rotacion: " << rotVar << std::endl;
        _applyFourier(MDaRefRot,MDaRefRotF); // fourier --> F2_r
        // computing relative traslation of rotated reference
        ccMatrix(MDaInF, MDaRefRotF, ccMatrixShift);// cross-correlation matrix
        maxByColumn(ccMatrixShift, ccVectorTx, Ydim, Xdim); // ccvMatrix to ccVector
        getShift(axTx, ccVectorTx,tx,Xdim);
        tx = -1. * tx;
        maxByRow(ccMatrixShift, ccVectorTy, Ydim, Xdim); // ccvMatrix to ccVector
        getShift(axTy, ccVectorTy,ty,Ydim);
        ty = -1. * ty;

        //*********** when strict, after first loop ***************
        // posible shifts
        vTx.push_back(tx);
        vTx.push_back(tx+0.5);
        vTx.push_back(tx-0.5);
        vTy.push_back(ty);
        vTy.push_back(ty+0.5);
        vTy.push_back(ty-0.5);

        for(int j = 0; j < 3; j++){
            for (int k = 0; k < 3; k++){
                // translate rotated version of MDaRef
                _applyShift(MDaRefRot, vTx[j], vTy[k], MDaRefRotShift);
                // Pearson coeff
                //        pearsonCorr(MDaIn, MDaRef, tempCoef);
                //        std::cout << "myCorr(f1,f2): " << tempCoef << std::endl;
                //        pearsonCorr(MDaIn, MDaRefRot, tempCoef);
                //        std::cout << "myCorr(f1,f2_r): " << tempCoef << std::endl;
                pearsonCorr(MDaIn, MDaRefRotShift, tempCoeff);
                //        std::cout << "myCorr(f1,f2_rt): " << tempCoef << std::endl;
                if ( tempCoeff > *(bestCoeff) ){
                    *(bestCoeff) = tempCoeff;
                    *(shift_x) = vTx[j];
                    *(shift_y) = vTy[k];
                    *(bestCandRot) = rotVar;
                }
            }
        }


    }

}

/* apply rotation */
void ProgAngularAssignmentMag::_applyRotationAndShift(MultidimArray<double> &MDaRef, double &rot, double &tx, double &ty,
                                              MultidimArray<double> &MDaRefRot){
    // Transform matrix
    Matrix2D<double> A(3,3);
    A.initIdentity();
    double ang, cosine, sine;
    ang = DEG2RAD(rot);
    cosine = cos(ang);
    sine = sin(ang);

    // rotation
    MAT_ELEM(A,0, 0) = cosine;
    MAT_ELEM(A,0, 1) = sine;
    MAT_ELEM(A,1, 0) = -sine;
    MAT_ELEM(A,1, 1) = cosine;

    // Shift
    MAT_ELEM(A,0, 2) = tx;
    MAT_ELEM(A,1, 2) = ty;

    //    applyGeometry(LINEAR, Mref, proj_ref[refno], A, IS_NOT_INV, DONT_WRAP);
    applyGeometry(LINEAR, MDaRefRot, MDaRef, A, IS_NOT_INV, DONT_WRAP);

}
