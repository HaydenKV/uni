#include <cassert>
#include <filesystem>

#define vtkRenderingContext2D_AUTOINIT 1(vtkRenderingContextOpenGL2)
#define vtkRenderingCore_AUTOINIT 3(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingOpenGL2)
#define vtkRenderingOpenGL2_AUTOINIT 1(vtkRenderingGL2PSOpenGL2)

#include <vtkAxis.h>
#include <vtkBrush.h>
#include <vtkBMPWriter.h>
#include <vtkChartLegend.h>
#include <vtkChartMatrix.h>
#include <vtkChartXY.h>
#include <vtkContextScene.h>
#include <vtkContextView.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkImageWriter.h>
#include <vtkJPEGWriter.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPen.h>
#include <vtkPlot.h>
#include <vtkPlotArea.h>
#include <vtkPlotFunctionalBag.h>
#include <vtkPlotPoints.h>
#include <vtkPNGWriter.h>
#include <vtkPNMWriter.h>
#include <vtkPostScriptWriter.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTable.h>
#include <vtkTextProperty.h>
#include <vtkTIFFWriter.h>
#include <vtkWindowToImageFilter.h>

#include "ballistic_plot.h"

void WriteImage(std::filesystem::path path, vtkRenderWindow * renWin, bool rgba)
{
    if (path.empty())
    {
        std::cerr << "No path provided." << std::endl;
        return;
    }
    else
    {
        std::string ext = path.extension().string();
        if (ext.empty())
        {
            ext = ".png";
            path = path / ".png";
        }
        std::locale loc;
        std::transform(ext.begin(), ext.end(), ext.begin(),
           [=](char const & c) { return std::tolower(c, loc); });
        auto writer = vtkSmartPointer<vtkImageWriter>::New();
        if (ext == ".bmp")
        {
            writer = vtkSmartPointer<vtkBMPWriter>::New();
        }
        else if (ext == ".jpg")
        {
            writer = vtkSmartPointer<vtkJPEGWriter>::New();
        }
        else if (ext == ".pnm")
        {
            writer = vtkSmartPointer<vtkPNMWriter>::New();
        }
        else if (ext == ".ps")
        {
            rgba = false;
            writer = vtkSmartPointer<vtkPostScriptWriter>::New();
        }
        else if (ext == ".tiff")
        {
            writer = vtkSmartPointer<vtkTIFFWriter>::New();
        }
        else
        {
            writer = vtkSmartPointer<vtkPNGWriter>::New();
        }

        vtkNew<vtkWindowToImageFilter> window_to_image_filter;
        window_to_image_filter->SetInput(renWin);
        window_to_image_filter->SetScale(1); // image quality
        if (rgba)
        {
            window_to_image_filter->SetInputBufferTypeToRGBA();
        }
        else
        {
            window_to_image_filter->SetInputBufferTypeToRGB();
        }
        // Read from the front buffer.
        window_to_image_filter->ReadFrontBufferOff();
        window_to_image_filter->Update();

        writer->SetFileName(path.string().c_str());
        writer->SetInputConnection(window_to_image_filter->GetOutputPort());
        writer->Write();
    }
}

void plot_simulation(
    const Eigen::VectorXd & t_hist, 
    const Eigen::MatrixXd & x_hist, 
    const Eigen::MatrixXd & mu_hist, 
    const Eigen::MatrixXd & sigma_hist)
{
    int nsteps  = x_hist.cols();
    int nx      = 3;
    assert(x_hist.rows() == nx);
    assert(mu_hist.rows() == nx);

    assert(t_hist.size() == nsteps);
    assert(mu_hist.cols() == nsteps);
    assert(sigma_hist.cols() == nsteps);

    // Sigma for the 99.7% confidence region
    double sigma  = 3;

    // Font size, colours, and linewidth
    // -----------------------------------------------
    auto title_fontsize   = 24;
    auto axis_fontsize    = 18;
    auto label_fontsize   = 18;
    auto legend_fontsize  = 18;
    auto linewidth        = 1.0;
    auto title_fontcolour  = "black";
    auto axis_fontcolour   = "black";
    auto label_fontcolour  = "black";

    // -----------------------------------------------
    // -----------------------------------------------
    // Instantiate the render stuff
    // -----------------------------------------------
    // -----------------------------------------------

    vtkNew<vtkContextView> view;
    vtkRenderWindow *renWin = view->GetRenderWindow();
    renWin->SetSize(1024, /*768*/ 1152);

    vtkNew<vtkNamedColors> colors;

    // -----------------------------------------------
    // -----------------------------------------------
    // Setup the chart matrix (similar to sub plots
    //   in matlab)
    // -----------------------------------------------
    // -----------------------------------------------

    vtkNew<vtkChartMatrix> matrix;
    view->GetScene()->AddItem(matrix);
    matrix->SetSize(vtkVector2i(2, 3));
    matrix->SetGutter(vtkVector2f(100.0, 100.0));
    matrix->SetBorders(100, 50, 50, 50);

    // -----------------------------------------------
    // -----------------------------------------------
    // Create the charts.
    // -----------------------------------------------
    // -----------------------------------------------

    // Top left (Height vs Time)
    // -----------------------------------------------
    vtkChart *topLeftChart = matrix->GetChart(vtkVector2i(0, 2));

    // Background
    topLeftChart->GetBackgroundBrush()->SetColorF(colors->GetColor3d("White").GetData());
    topLeftChart->GetBackgroundBrush()->SetOpacityF(0.4);

    // Title
    //topLeftChart->SetTitle("State estimates");
    topLeftChart->GetTitleProperties()->SetFontSize(title_fontsize);
    topLeftChart->GetTitleProperties()->SetColor(colors->GetColor3d(title_fontcolour).GetData());

    // X axis
    auto xAxis = topLeftChart->GetAxis(vtkAxis::BOTTOM);
    xAxis->GetGridPen()->SetColor(200, 200, 200, 255);
    xAxis->SetTitle("Time [s]");
    xAxis->GetTitleProperties()->SetFontSize(axis_fontsize);
    xAxis->GetTitleProperties()->SetColor(colors->GetColor3d(axis_fontcolour).GetData());
    xAxis->GetLabelProperties()->SetFontSize(label_fontsize);
    xAxis->GetLabelProperties()->SetColor(colors->GetColor3d(label_fontcolour).GetData());

    // Y axis
    auto yAxis = topLeftChart->GetAxis(vtkAxis::LEFT);
    yAxis->GetGridPen()->SetColor(200, 200, 200, 255);
    yAxis->SetTitle("Height [m]");
    yAxis->GetTitleProperties()->SetFontSize(axis_fontsize);
    yAxis->GetTitleProperties()->SetColor(colors->GetColor3d(axis_fontcolour).GetData());
    yAxis->GetLabelProperties()->SetFontSize(label_fontsize);
    yAxis->GetLabelProperties()->SetColor(colors->GetColor3d(label_fontcolour).GetData());

    // Top right (Height standard deviation vs Time)
    // -----------------------------------------------
    vtkChart *topRightChart = matrix->GetChart(vtkVector2i(1, 2));

    // Background
    topRightChart->GetBackgroundBrush()->SetColorF(colors->GetColor3d("White").GetData());
    topRightChart->GetBackgroundBrush()->SetOpacityF(0.4);

    // Title
    topRightChart->SetTitle("Marginal standard deviations");
    topRightChart->GetTitleProperties()->SetFontSize(title_fontsize);
    topRightChart->GetTitleProperties()->SetColor(colors->GetColor3d(title_fontcolour).GetData());

    // X axis
    xAxis = topRightChart->GetAxis(vtkAxis::BOTTOM);
    xAxis->GetGridPen()->SetColor(200, 200, 200, 255);
    xAxis->SetTitle("Time [s]");
    xAxis->GetTitleProperties()->SetFontSize(axis_fontsize);
    xAxis->GetTitleProperties()->SetColor(colors->GetColor3d(axis_fontcolour).GetData());
    xAxis->GetLabelProperties()->SetFontSize(label_fontsize);
    xAxis->GetLabelProperties()->SetColor(colors->GetColor3d(label_fontcolour).GetData());

    // Y axis
    yAxis = topRightChart->GetAxis(vtkAxis::LEFT);
    yAxis->GetGridPen()->SetColor(200, 200, 200, 255);
    yAxis->SetTitle("Height [m]");
    yAxis->GetTitleProperties()->SetFontSize(axis_fontsize);
    yAxis->GetTitleProperties()->SetColor(colors->GetColor3d(axis_fontcolour).GetData());
    yAxis->GetLabelProperties()->SetFontSize(label_fontsize);
    yAxis->GetLabelProperties()->SetColor(colors->GetColor3d(label_fontcolour).GetData());
    yAxis->SetLogScale(true);

    // Middle left (Velocity vs Time)
    // -----------------------------------------------
    vtkChart *middleLeftChart = matrix->GetChart(vtkVector2i(0, 1));

    // Background
    middleLeftChart->GetBackgroundBrush()->SetColorF(colors->GetColor3d("White").GetData());
    middleLeftChart->GetBackgroundBrush()->SetOpacityF(0.4);
    
    // X axis
    xAxis = middleLeftChart->GetAxis(vtkAxis::BOTTOM);
    xAxis->GetGridPen()->SetColor(200, 200, 200, 255);
    xAxis->SetTitle("Time [s]");
    xAxis->GetTitleProperties()->SetFontSize(axis_fontsize);
    xAxis->GetTitleProperties()->SetColor(colors->GetColor3d(axis_fontcolour).GetData());
    xAxis->GetLabelProperties()->SetFontSize(label_fontsize);
    xAxis->GetLabelProperties()->SetColor(colors->GetColor3d(label_fontcolour).GetData());

    // Y axis
    yAxis = middleLeftChart->GetAxis(vtkAxis::LEFT);
    yAxis->GetGridPen()->SetColor(200, 200, 200, 255);
    yAxis->SetTitle("Velocity [m/s]");
    yAxis->GetTitleProperties()->SetFontSize(axis_fontsize);
    yAxis->GetTitleProperties()->SetColor(colors->GetColor3d(axis_fontcolour).GetData());
    yAxis->GetLabelProperties()->SetFontSize(label_fontsize);
    yAxis->GetLabelProperties()->SetColor(colors->GetColor3d(label_fontcolour).GetData());

    // Middle right (Velocity vs Time)
    // -----------------------------------------------
    vtkChart *middleRightChart = matrix->GetChart(vtkVector2i(1, 1));

    // Background
    middleRightChart->GetBackgroundBrush()->SetColorF(colors->GetColor3d("White").GetData());
    middleRightChart->GetBackgroundBrush()->SetOpacityF(0.4);

    // X axis
    xAxis = middleRightChart->GetAxis(vtkAxis::BOTTOM);
    xAxis->GetGridPen()->SetColor(200, 200, 200, 255);
    xAxis->SetTitle("Time [s]");
    xAxis->GetTitleProperties()->SetFontSize(axis_fontsize);
    xAxis->GetTitleProperties()->SetColor(colors->GetColor3d(axis_fontcolour).GetData());
    xAxis->GetLabelProperties()->SetFontSize(label_fontsize);
    xAxis->GetLabelProperties()->SetColor(colors->GetColor3d(label_fontcolour).GetData());

    // Y axis
    yAxis = middleRightChart->GetAxis(vtkAxis::LEFT);
    yAxis->GetGridPen()->SetColor(200, 200, 200, 255);
    yAxis->SetTitle("Velocity [m/s]");
    yAxis->GetTitleProperties()->SetFontSize(axis_fontsize);
    yAxis->GetTitleProperties()->SetColor(colors->GetColor3d(axis_fontcolour).GetData());
    yAxis->GetLabelProperties()->SetFontSize(label_fontsize);
    yAxis->GetLabelProperties()->SetColor(colors->GetColor3d(label_fontcolour).GetData());
    yAxis->SetLogScale(true);

    // Bottom left (Ballistic coefficient vs Time)
    // -----------------------------------------------
    vtkChart *bottomLeftChart = matrix->GetChart(vtkVector2i(0, 0));
    
    /*
     *          __.--,
     *     ,--'~-.,;=/
     *    {  /,”” ‘ |
     *     `|\_ (@\(@\_
     *      \(  _____(./
     *      | `-~--~-‘\
     *      | /   /  , \
     *      | |   (\  \  \
     *     _\_|.-‘` `’| ,-= )
     *    (_  ,\ ____  \  ,/
     *      \  |--===--(,,/
     *       ```========-/
     *         \_______ /
     *  SOMEBODY TOUCHA MY SPAGHET!
     */ 

    // Background
    bottomLeftChart->GetBackgroundBrush()->SetColorF(colors->GetColor3d("White").GetData());
    bottomLeftChart->GetBackgroundBrush()->SetOpacityF(0.4);

    // Title
    //bottomLeftChart->SetTitle("State estimates");
    bottomLeftChart->GetTitleProperties()->SetFontSize(title_fontsize);
    bottomLeftChart->GetTitleProperties()->SetColor(colors->GetColor3d(title_fontcolour).GetData());

    // X axis
    xAxis = bottomLeftChart->GetAxis(vtkAxis::BOTTOM);
    xAxis->GetGridPen()->SetColor(200, 200, 200, 255);
    xAxis->SetTitle("Time [s]");
    xAxis->GetTitleProperties()->SetFontSize(axis_fontsize);
    xAxis->GetTitleProperties()->SetColor(colors->GetColor3d(axis_fontcolour).GetData());
    xAxis->GetLabelProperties()->SetFontSize(label_fontsize);
    xAxis->GetLabelProperties()->SetColor(colors->GetColor3d(label_fontcolour).GetData());

    // Y axis
    yAxis = bottomLeftChart->GetAxis(vtkAxis::LEFT);
    yAxis->GetGridPen()->SetColor(200, 200, 200, 255);
    yAxis->SetTitle("Ballistic coefficient (m²/kg)");
    yAxis->GetTitleProperties()->SetFontSize(axis_fontsize);
    yAxis->GetTitleProperties()->SetColor(colors->GetColor3d(axis_fontcolour).GetData());
    yAxis->GetLabelProperties()->SetFontSize(label_fontsize);
    yAxis->GetLabelProperties()->SetColor(colors->GetColor3d(label_fontcolour).GetData());


    // Bottom right (Ballistic coefficient standard deviation vs Time)
    // -----------------------------------------------
    vtkChart *bottomRightChart = matrix->GetChart(vtkVector2i(1, 0));

    /*
     *          __.--,
     *     ,--'~-.,;=/
     *    {  /,”” ‘ |
     *     `|\_ (@\(@\_
     *      \(  _____(./
     *      | `-~--~-‘\
     *      | /   /  , \
     *      | |   (\  \  \
     *     _\_|.-‘` `’| ,-= )
     *    (_  ,\ ____  \  ,/
     *      \  |--===--(,,/
     *       ```========-/
     *         \_______ /
     *  SOMEBODY TOUCHA MY SPAGHET!
     */ 

    // Background
    bottomRightChart->GetBackgroundBrush()->SetColorF(colors->GetColor3d("White").GetData());
    bottomRightChart->GetBackgroundBrush()->SetOpacityF(0.4);

    // Title
    bottomRightChart->SetTitle("Marginal standard deviations");
    bottomRightChart->GetTitleProperties()->SetFontSize(title_fontsize);
    bottomRightChart->GetTitleProperties()->SetColor(colors->GetColor3d(title_fontcolour).GetData());

    // X axis
    xAxis = bottomRightChart->GetAxis(vtkAxis::BOTTOM);
    xAxis->GetGridPen()->SetColor(200, 200, 200, 255);
    xAxis->SetTitle("Time [s]");
    xAxis->GetTitleProperties()->SetFontSize(axis_fontsize);
    xAxis->GetTitleProperties()->SetColor(colors->GetColor3d(axis_fontcolour).GetData());
    xAxis->GetLabelProperties()->SetFontSize(label_fontsize);
    xAxis->GetLabelProperties()->SetColor(colors->GetColor3d(label_fontcolour).GetData());

    // Y axis
    yAxis = bottomRightChart->GetAxis(vtkAxis::LEFT);
    yAxis->GetGridPen()->SetColor(200, 200, 200, 255);
    yAxis->SetTitle("Ballistic coefficient (m²/kg)");
    yAxis->GetTitleProperties()->SetFontSize(axis_fontsize);
    yAxis->GetTitleProperties()->SetColor(colors->GetColor3d(axis_fontcolour).GetData());
    yAxis->GetLabelProperties()->SetFontSize(label_fontsize);
    yAxis->GetLabelProperties()->SetColor(colors->GetColor3d(label_fontcolour).GetData());
    yAxis->SetLogScale(true);

    // -----------------------------------------------
    // -----------------------------------------------
    // Create data table
    // -----------------------------------------------
    // -----------------------------------------------
    vtkNew<vtkTable> table;

    vtkNew<vtkFloatArray> 
        arr_t,
        arr_height_true,
        arr_height_est,
        arr_velocity_true,
        arr_velocity_est,
        arr_bcoeff_true,
        arr_bcoeff_est,
        arr_mu1_plus_sigma1,
        arr_mu1_minus_sigma1,
        arr_mu2_plus_sigma2,
        arr_mu2_minus_sigma2,
        arr_mu3_plus_sigma3,
        arr_mu3_minus_sigma3,
        arr_sigma1,
        arr_sigma2,
        arr_sigma3;

    //  Initialise the table
    // -----------------------------------------------
    enum
    {
        // Means
        TABLE_TIME,
        TABLE_HEIGHT_TRUE,
        TABLE_HEIGHT_EST,
        TABLE_VELOCITY_TRUE,
        TABLE_VELOCITY_EST,
        TABLE_BCOEFF_TRUE,
        TABLE_BCOEFF_EST,

        // Confidence region
        TABLE_MU1_PLUS_SIGMA1,
        TABLE_MU1_MINUS_SIGMA1,
        TABLE_MU2_PLUS_SIGMA2,
        TABLE_MU2_MINUS_SIGMA2,
        TABLE_MU3_PLUS_SIGMA3,
        TABLE_MU3_MINUS_SIGMA3,

        // Standard deviations
        TABLE_SIGMA1,
        TABLE_SIGMA2,
        TABLE_SIGMA3,
    };

    #define KEY(NAME) "KEY_" #NAME

    // Make columns of the table
    arr_t->SetName(KEY(TABLE_TIME)); table->AddColumn(arr_t);
    
    arr_height_true->SetName(KEY(TABLE_HEIGHT_TRUE)); table->AddColumn(arr_height_true);
    arr_height_est->SetName(KEY(TABLE_HEIGHT_EST)); table->AddColumn(arr_height_est);
    
    arr_velocity_true->SetName(KEY(TABLE_VELOCITY_TRUE)); table->AddColumn(arr_velocity_true);
    arr_velocity_est->SetName(KEY(TABLE_VELOCITY_EST)); table->AddColumn(arr_velocity_est);
    
    arr_bcoeff_true->SetName(KEY(TABLE_BCOEFF_TRUE)); table->AddColumn(arr_bcoeff_true);
    arr_bcoeff_est->SetName(KEY(TABLE_BCOEFF_EST)); table->AddColumn(arr_bcoeff_est);
    
    arr_mu1_plus_sigma1->SetName(KEY(TABLE_MU1_PLUS_SIGMA1)); table->AddColumn(arr_mu1_plus_sigma1);
    arr_mu1_minus_sigma1->SetName(KEY(TABLE_MU1_MINUS_SIGMA1)); table->AddColumn(arr_mu1_minus_sigma1);

    arr_mu2_plus_sigma2->SetName(KEY(TABLE_MU2_PLUS_SIGMA2)); table->AddColumn(arr_mu2_plus_sigma2);
    arr_mu2_minus_sigma2->SetName(KEY(TABLE_MU2_MINUS_SIGMA2)); table->AddColumn(arr_mu2_minus_sigma2);

    arr_mu3_plus_sigma3->SetName(KEY(TABLE_MU3_PLUS_SIGMA3)); table->AddColumn(arr_mu3_plus_sigma3);
    arr_mu3_minus_sigma3->SetName(KEY(TABLE_MU3_MINUS_SIGMA3)); table->AddColumn(arr_mu3_minus_sigma3);

    arr_sigma1->SetName(KEY(TABLE_SIGMA1)); table->AddColumn(arr_sigma1);
    arr_sigma2->SetName(KEY(TABLE_SIGMA2)); table->AddColumn(arr_sigma2);
    arr_sigma3->SetName(KEY(TABLE_SIGMA3)); table->AddColumn(arr_sigma3);

    //  Fill in the table with data from the simulation
    // -----------------------------------------------
    table->SetNumberOfRows(nsteps);

    for (int i = 0; i < nsteps; ++i)
    {
        table->SetValue(i,                  TABLE_TIME, t_hist(i));
        table->SetValue(i,           TABLE_HEIGHT_TRUE, x_hist(0,i));
        table->SetValue(i,         TABLE_VELOCITY_TRUE, x_hist(1,i));
        table->SetValue(i,           TABLE_BCOEFF_TRUE, x_hist(2,i));

        table->SetValue(i,            TABLE_HEIGHT_EST, mu_hist(0,i));
        table->SetValue(i,          TABLE_VELOCITY_EST, mu_hist(1,i));
        table->SetValue(i,            TABLE_BCOEFF_EST, mu_hist(2,i));

        table->SetValue(i,       TABLE_MU1_PLUS_SIGMA1, mu_hist(0,i) + sigma*sigma_hist(0,i));
        table->SetValue(i,      TABLE_MU1_MINUS_SIGMA1, mu_hist(0,i) - sigma*sigma_hist(0,i));
        table->SetValue(i,       TABLE_MU2_PLUS_SIGMA2, mu_hist(1,i) + sigma*sigma_hist(1,i));
        table->SetValue(i,      TABLE_MU2_MINUS_SIGMA2, mu_hist(1,i) - sigma*sigma_hist(1,i));
        table->SetValue(i,       TABLE_MU3_PLUS_SIGMA3, mu_hist(2,i) + sigma*sigma_hist(2,i));
        table->SetValue(i,      TABLE_MU3_MINUS_SIGMA3, mu_hist(2,i) - sigma*sigma_hist(2,i));

        table->SetValue(i,                TABLE_SIGMA1, sigma_hist(0,i));
        table->SetValue(i,                TABLE_SIGMA2, sigma_hist(1,i));
        table->SetValue(i,                TABLE_SIGMA3, sigma_hist(2,i));
    }

    // -----------------------------------------------
    // -----------------------------------------------
    // Plot things
    // -----------------------------------------------
    // -----------------------------------------------
    vtkPlot* line;
    vtkPlotArea* area;

    // Top left plot (Height vs Time)
    // -----------------------------------------------
    
    // true value
    line = topLeftChart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, TABLE_HEIGHT_TRUE);
    line->SetColor(255, 0, 0, 255); // Red for true values
    line->SetWidth(linewidth);
    line->SetLabel("true");

    // mean value
    line = topLeftChart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, TABLE_HEIGHT_EST);
    line->SetColor(0, 0, 255, 255); // Blue for estimated values
    line->SetWidth(linewidth);
    line->SetLabel("mean");

    // confidence region
    area = dynamic_cast<vtkPlotArea *>(topLeftChart->AddPlot(vtkChart::AREA));
    area->SetInputData(table);
    area->SetInputArray(0, KEY(TABLE_TIME));
    area->SetInputArray(1, KEY(TABLE_MU1_PLUS_SIGMA1));
    area->SetInputArray(2, KEY(TABLE_MU1_MINUS_SIGMA1));
    area->GetBrush()->SetColorF(0, 0, 1, 0.1); // Blue with low opacity                            
    area->SetLabel("99.7% CR");

    // Show legend
    topLeftChart->SetShowLegend(true);
    topLeftChart->GetLegend()->SetHorizontalAlignment(vtkChartLegend::RIGHT);
    topLeftChart->GetLegend()->SetVerticalAlignment(vtkChartLegend::TOP);
    topLeftChart->GetLegend()->SetLabelSize(legend_fontsize);
    
    // Top right plot (Height standard deviation vs Time)
    // -----------------------------------------------
    
    // Plot
    line = topRightChart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, TABLE_SIGMA1);
    line->SetColor(0, 0, 255, 255);
    line->SetWidth(linewidth);

    topRightChart->SetShowLegend(false);

    // Middle left plot (Velocity vs Time)
    // -----------------------------------------------

    // true value
    line = middleLeftChart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, TABLE_VELOCITY_TRUE);
    line->SetColor(255, 0, 0, 255); // Red for true values
    line->SetWidth(linewidth);
    line->SetLabel("true");

    // mean value
    line = middleLeftChart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, TABLE_VELOCITY_EST);
    line->SetColor(0, 0, 255, 255); // Blue for estimated values
    line->SetWidth(linewidth);    
    line->SetLabel("mean");

    // confidence region
    area = dynamic_cast<vtkPlotArea *>(middleLeftChart->AddPlot(vtkChart::AREA));
    area->SetInputData(table);
    area->SetInputArray(0, KEY(TABLE_TIME));
    area->SetInputArray(1, KEY(TABLE_MU2_PLUS_SIGMA2));
    area->SetInputArray(2, KEY(TABLE_MU2_MINUS_SIGMA2));
    area->GetBrush()->SetColorF(0, 0, 1, 0.1); // Blue with low opacity
    area->SetLabel("99.7% CR");

    // Show legend
    middleLeftChart->SetShowLegend(true);
    middleLeftChart->GetLegend()->SetHorizontalAlignment(vtkChartLegend::RIGHT);
    middleLeftChart->GetLegend()->SetVerticalAlignment(vtkChartLegend::BOTTOM);
    middleLeftChart->GetLegend()->SetLabelSize(legend_fontsize);

    // Middle right plot (Velocity vs Time)
    // -----------------------------------------------

    // Plot
    line = middleRightChart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, TABLE_SIGMA2);
    line->SetColor(0, 0, 255, 255);
    line->SetWidth(linewidth);

    middleRightChart->SetShowLegend(false);

    // Bottom left plot (Ballistic coeff vs Time)
    // -----------------------------------------------

    /*
     *          __.--,
     *     ,--'~-.,;=/
     *    {  /,”” ‘ |
     *     `|\_ (@\(@\_
     *      \(  _____(./
     *      | `-~--~-‘\
     *      | /   /  , \
     *      | |   (\  \  \
     *     _\_|.-‘` `’| ,-= )
     *    (_  ,\ ____  \  ,/
     *      \  |--===--(,,/
     *       ```========-/
     *         \_______ /
     *  SOMEBODY TOUCHA MY SPAGHET!
     */ 
    // Bottom left plot (Ballistic coeff vs Time)
    // true value
    line = bottomLeftChart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, TABLE_BCOEFF_TRUE);
    line->SetColor(255, 0, 0, 255); // Red for true values
    line->SetWidth(linewidth);
    line->SetLabel("true");

    // mean value
    line = bottomLeftChart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, TABLE_BCOEFF_EST);
    line->SetColor(0, 0, 255, 255); // Blue for estimated values
    line->SetWidth(linewidth);
    line->SetLabel("mean");

    // confidence region (99.7%)
    area = dynamic_cast<vtkPlotArea *>(bottomLeftChart->AddPlot(vtkChart::AREA));
    area->SetInputData(table);
    area->SetInputArray(0, KEY(TABLE_TIME));
    area->SetInputArray(1, KEY(TABLE_MU3_PLUS_SIGMA3));
    area->SetInputArray(2, KEY(TABLE_MU3_MINUS_SIGMA3));
    area->GetBrush()->SetColorF(0, 0, 1, 0.1); // Blue with low opacity
    area->SetLabel("99.7% CR");

    // Show legend
    bottomLeftChart->SetShowLegend(true);
    bottomLeftChart->GetLegend()->SetHorizontalAlignment(vtkChartLegend::RIGHT);
    bottomLeftChart->GetLegend()->SetVerticalAlignment(vtkChartLegend::BOTTOM);
    bottomLeftChart->GetLegend()->SetLabelSize(legend_fontsize);


    // Bottom right plot (Ballistic coeff standard deviation vs Time)
    // -----------------------------------------------

    /*
     *          __.--,
     *     ,--'~-.,;=/
     *    {  /,”” ‘ |
     *     `|\_ (@\(@\_
     *      \(  _____(./
     *      | `-~--~-‘\
     *      | /   /  , \
     *      | |   (\  \  \
     *     _\_|.-‘` `’| ,-= )
     *    (_  ,\ ____  \  ,/
     *      \  |--===--(,,/
     *       ```========-/
     *         \_______ /
     *  SOMEBODY TOUCHA MY SPAGHET!
     */  
    // Bottom right plot (Ballistic coeff standard deviation vs Time)
    line = bottomRightChart->AddPlot(vtkChart::LINE);
    line->SetInputData(table, 0, TABLE_SIGMA3);
    line->SetColor(0, 0, 255, 255);
    line->SetWidth(linewidth);

    bottomRightChart->SetShowLegend(false);

    #undef KEY

    // Do the render
    // -----------------------------------------------
    view->GetRenderer()->SetBackground(colors->GetColor3d("White").GetData());
    renWin->SetMultiSamples(0);
    renWin->Render();
    renWin->SetWindowName("Ballistic state estimation");

    WriteImage("../data/Ballistic.png", renWin, false);

    vtkRenderWindowInteractor *iRen = view->GetInteractor();
    iRen->Initialize();
    iRen->Start();
}
