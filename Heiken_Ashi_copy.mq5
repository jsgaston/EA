//+------------------------------------------------------------------+
//|                                                      ProjectName |
//|                                      Copyright 2020, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                  Heiken_Ashi.mq5 |
//|                             Copyright 2000-2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2000-2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//--- indicator settings
#property indicator_chart_window
#property indicator_buffers 5
#property indicator_plots   1
#property indicator_type1   DRAW_COLOR_CANDLES
#property indicator_color1  DodgerBlue, Red
#property indicator_label1  "Heiken Ashi Open;Heiken Ashi High;Heiken Ashi Low;Heiken Ashi Close"
//--- indicator buffers
double ExtOBuffer[];
double ExtHBuffer[];
double ExtLBuffer[];
double ExtCBuffer[];
double ExtColorBuffer[];
// Declaración de la matriz

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
void OnInit()
  {
//--- indicator buffers mapping
   SetIndexBuffer(0,ExtOBuffer,INDICATOR_DATA);
   SetIndexBuffer(1,ExtHBuffer,INDICATOR_DATA);
   SetIndexBuffer(2,ExtLBuffer,INDICATOR_DATA);
   SetIndexBuffer(3,ExtCBuffer,INDICATOR_DATA);
   SetIndexBuffer(4,ExtColorBuffer,INDICATOR_COLOR_INDEX);
//---
   IndicatorSetInteger(INDICATOR_DIGITS,_Digits);
//--- sets first bar from what index will be drawn
   IndicatorSetString(INDICATOR_SHORTNAME,"Heiken Ashi");
//--- sets drawing line empty value
   PlotIndexSetDouble(0,PLOT_EMPTY_VALUE,0.0);
  }

//+------------------------------------------------------------------+
//| Heiken Ashi                                                      |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   int start;
//--- preliminary calculations
   if(prev_calculated==0)
     {
      ExtLBuffer[0]=low[0];
      ExtHBuffer[0]=high[0];
      ExtOBuffer[0]=open[0];
      ExtCBuffer[0]=close[0];
      start=1;
     }
   else
      start=prev_calculated-1;

//--- the main loop of calculations
   for(int i=start; i<rates_total && !IsStopped(); i++)
     {
      double ha_open =(ExtOBuffer[i-1]+ExtCBuffer[i-1])/2;
      double ha_close=(open[i]+high[i]+low[i]+close[i])/4;
      double ha_high =MathMax(high[i],MathMax(ha_open,ha_close));
      double ha_low  =MathMin(low[i],MathMin(ha_open,ha_close));

      ExtLBuffer[i]=ha_low;
      ExtHBuffer[i]=ha_high;
      ExtOBuffer[i]=ha_open;
      ExtCBuffer[i]=ha_close;

      //--- set candle color
      if(ha_open<ha_close)
         ExtColorBuffer[i]=0.0; // set color DodgerBlue
      else
         ExtColorBuffer[i]=1.0; // set color Red
     }

//---
   return(rates_total);

// The MQL5 language does not have a built-in function for returning multiple values. However, you can return multiple values by using a structure or array. For example, you could create a structure that contains the rate and color values, and then return the structure. with the calculated values.

//The MQL5 language provides a variety of functions and objects that can be used to calculate rate and color values. These include the RateInfo object, which can be used to retrieve rate information such as the current bid and ask prices, as well as the Color object, which can be used to set the color of a chart element. Additionally, the Math and MathEx functions can be used to perform mathematical calculations, such as calculating the average of a set of values. Finally, the ChartObjects functions can be used to create and manipulate chart objects, such as lines and arrows.


  }
//+------------------------------------------------------------------+
// Función que devuelve una estructura que contiene el array
