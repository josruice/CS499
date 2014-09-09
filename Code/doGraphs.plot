# Create PDF images as output.
# usage: gnuplot doGraphs.plot

# General output settings.
set terminal postscript enhanced color font 'Helvetica,18' linewidth 2
set encoding default

set ylabel "% accuracy"
set xlabel "# clusters"
set mxtics
set mytics
set yrange [1:100]
# set key auto

NAME = "../Figures/numberOfClustersTestedReal"
FILENAME = "../Results/plotableNumClusters.txt"
set title "Material classification accuracy tested with real data w.r.t. number of clusters."
set key at 950, 30
#set autoscale y
#set yrange [0.997:1]

set output "| ps2pdf - ".NAME.".pdf"
plot FILENAME using 1:3 \
    title "NB - Trained with predicted data" lt 1 lc rgb "#DD0000" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 1:5 \
    title "NB - Trained with real data" lt 1 lc rgb "#DD8888" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 1:7 \
    title "SVM - Trained with predicted data" lt 1 lc rgb "#0000DD" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 1:9 \
    title "SVM - Trained with real data" lt 1 lc rgb "#8888DD" lw 2 with lines



NAME = "../Figures/numberOfClustersTestedPred"
FILENAME = "../Results/plotableNumClusters.txt"
set title "Material classification accuracy tested with predicted data w.r.t. number of clusters."
set key at 950, 90
#set autoscale y
#set yrange [0.997:1]

set output "| ps2pdf - ".NAME.".pdf"
plot FILENAME using 1:2 \
    title "NB - Trained with predicted data" lt 1 lc rgb "#DD0000" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 1:4 \
    title "NB - Trained with real data" lt 1 lc rgb "#DD8888" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 1:6 \
    title "SVM - Trained with predicted data" lt 1 lc rgb "#0000DD" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 1:8 \
    title "SVM - Trained with real data" lt 1 lc rgb "#8888DD" lw 2 with lines







NAME = "../Figures/lambdaTestedReal"
FILENAME = "../Results/plotableLambda.txt"
set title "Material classification accuracy tested with real data w.r.t. value of SVMs lambda."
set key at 0.05, 30
set xrange [0.000001:0.1]
set logscale x
set xlabel "lambda value"
#set autoscale y
#set yrange [0.997:1]

set output "| ps2pdf - ".NAME.".pdf"
plot FILENAME using 1:3 \
    title "NB - Trained with predicted data" lt 1 lc rgb "#DD0000" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 1:5 \
    title "NB - Trained with real data" lt 1 lc rgb "#DD8888" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 1:7 \
    title "SVM - Trained with predicted data" lt 1 lc rgb "#0000DD" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 1:9 \
    title "SVM - Trained with real data" lt 1 lc rgb "#8888DD" lw 2 with lines



NAME = "../Figures/lambdaTestedPred"
FILENAME = "../Results/plotableLambda.txt"
set title "Material classification accuracy tested with predicted data w.r.t. value of SVMs lambda."
set key at 0.05, 90
#set autoscale y
#set yrange [0.997:1]

set output "| ps2pdf - ".NAME.".pdf"
plot FILENAME using 1:2 \
    title "NB - Trained with predicted data" lt 1 lc rgb "#DD0000" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 1:4 \
    title "NB - Trained with real data" lt 1 lc rgb "#DD8888" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 1:6 \
    title "SVM - Trained with predicted data" lt 1 lc rgb "#0000DD" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 1:8 \
    title "SVM - Trained with real data" lt 1 lc rgb "#8888DD" lw 2 with lines







NAME = "../Figures/descriptorsTestedReal"
FILENAME = "../Results/plotableDescriptors.txt"
set title "Material classification accuracy tested with real data w.r.t. # SIFT descriptors."
set key at 3, 30
# set logscale x
set xlabel "Feature method and # maximum descriptors per image"
unset logscale x
set xrange [-0.25:3.25]

set output "| ps2pdf - ".NAME.".pdf"
plot FILENAME using 3:xtic(1) \
    title "NB - Trained with predicted data" lt 1 lc rgb "#DD0000" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 5:xtic(1) \
    title "NB - Trained with real data" lt 1 lc rgb "#DD8888" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 7:xtic(1) \
    title "SVM - Trained with predicted data" lt 1 lc rgb "#0000DD" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 9:xtic(1) \
    title "SVM - Trained with real data" lt 1 lc rgb "#8888DD" lw 2 with lines



NAME = "../Figures/descriptorsTestedPred"
FILENAME = "../Results/plotableDescriptors.txt"
set title "Material classification accuracy tested with predicted data w.r.t. # SIFT descriptors."
set key at 3, 90
#set autoscale y
#set yrange [0.997:1]

set output "| ps2pdf - ".NAME.".pdf"
plot FILENAME using 2:xtic(1) \
    title "NB - Trained with predicted data" lt 1 lc rgb "#DD0000" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 4:xtic(1) \
    title "NB - Trained with real data" lt 1 lc rgb "#DD8888" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 6:xtic(1) \
    title "SVM - Trained with predicted data" lt 1 lc rgb "#0000DD" lw 2 with lines

set output "| ps2pdf - ".NAME.".pdf"
replot FILENAME using 8:xtic(1) \
    title "SVM - Trained with real data" lt 1 lc rgb "#8888DD" lw 2 with lines