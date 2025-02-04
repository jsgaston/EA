//+------------------------------------------------------------------+
//|                                                      Heiken1.mq5 |
//|              Copyright 2024, Javier S. Gastón de Iriarte Cabrera |
//|                      https://www.mql5.com/en/users/jsgaston/news |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Javier S. Gastón de Iriarte Cabrera"
#property link      "https://www.mql5.com/en/users/jsgaston/news"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
CTrade trade;

// HA colors
color hacolor;

// Calculation HA Values
double haopen[];
double haclose[];
double hahigh[];
double halow[];

input ENUM_TIMEFRAMES my_timeframe_adx = PERIOD_CURRENT;
input ENUM_TIMEFRAMES my_timeframe_Heiken = PERIOD_CURRENT;
input double InpLots = 0.1;            // Lotes para abrir posición
input bool InpUseStops = true;         // Use stops in trading
input int StopLossPips = 50;            // Stop Loss en pips
input int TakeProfitPips = 100;         // Take Profit en pips
input int VolumeThreshold = 4750;        // Umbral de volumen para confirmar la entrada
int ATRHandle;
int handle_iCustomHeiken;
int handle_volume;
double Sl = 0.0;
double Tp = 0.0;
int lastBarsCount = 0;
double prevHaclose = 0.0;
double prevHaopen = 0.0;

bool beforeGreen = false;
bool beforeRed = false;

int handle_adx;

int Order = 0;

input int ADXThreshold = 22;            // ADX Close
input int ADXThresholdOpen = 19;        // ADX Open

#define SAMPLE_SIZE 120

#resource "/Files/model.eurusd.D1.120.onnx" as uchar ExtModel[]

long     ExtHandle=INVALID_HANDLE;
int      ExtPredictedClass=-1;
datetime ExtNextBar=0;
datetime ExtNextDay=0;
float    ExtMin=0.0;
float    ExtMax=0.0;

double lastPredicted = 0.0;
double predicted = 0.0;

CTrade   ExtTrade;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
   handle_iCustomHeiken = iCustom(_Symbol, my_timeframe_Heiken, "\\Indicators\\Heiken_Ashi_copy");
   if(handle_iCustomHeiken == INVALID_HANDLE)
     {
      //--- tell about the failure and output the error code
      PrintFormat("Failed to create handle of the iCustom indicator for the symbol %s/%s, error code %d",
                  _Symbol,
                  EnumToString(my_timeframe_Heiken),
                  GetLastError());
      //--- the indicator is stopped early
      return(INIT_FAILED);
     }

   handle_adx = iADX(_Symbol, my_timeframe_adx, 14);
   if(handle_adx == INVALID_HANDLE)
     {
      PrintFormat("Failed to create handle of the iCustom indicator for the symbol %s/%s, error code %d",
                  _Symbol,
                  EnumToString(my_timeframe_adx),
                  GetLastError());
      return(INIT_FAILED);
     }

   handle_volume = iVolumes(_Symbol, my_timeframe_Heiken,VOLUME_TICK);
   if(handle_volume == INVALID_HANDLE)
     {
      PrintFormat("Failed to create handle of the Volume indicator for the symbol %s/%s, error code %d",
                  _Symbol,
                  EnumToString(my_timeframe_Heiken),
                  GetLastError());
      return(INIT_FAILED);
     }



//--- create a model from static buffer
   ExtHandle=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(ExtHandle==INVALID_HANDLE)
     {
      Print("OnnxCreateFromBuffer error ",GetLastError());
      return(INIT_FAILED);
     }

//--- since not all sizes defined in the input tensor we must set them explicitly
//--- first index - batch size, second index - series size, third index - number of series (only Close)
   const long input_shape[] = {1,SAMPLE_SIZE,1};
   if(!OnnxSetInputShape(ExtHandle,ONNX_DEFAULT,input_shape))
     {
      Print("OnnxSetInputShape error ",GetLastError());
      return(INIT_FAILED);
     }

//--- since not all sizes defined in the output tensor we must set them explicitly
//--- first index - batch size, must match the batch size of the input tensor
//--- second index - number of predicted prices (we only predict Close)
   const long output_shape[] = {1,1};
   if(!OnnxSetOutputShape(ExtHandle,0,output_shape))
     {
      Print("OnnxSetOutputShape error ",GetLastError());
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {

   if(ExtHandle!=INVALID_HANDLE)
     {
      OnnxRelease(ExtHandle);
      ExtHandle=INVALID_HANDLE;
     }
//--- Destroy the event timer
   EventKillTimer();
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

//--- check new day
   if(TimeCurrent()>=ExtNextDay)
     {
      GetMinMax();
      //--- set next day time
      ExtNextDay=TimeCurrent();
      ExtNextDay-=ExtNextDay%PeriodSeconds(PERIOD_D1);
      ExtNextDay+=PeriodSeconds(PERIOD_D1);
     }

//--- check new bar
   if(TimeCurrent()<ExtNextBar)
      return;
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float close=(float)iClose(_Symbol,_Period,0);
   if(ExtMin>close)
      ExtMin=close;
   if(ExtMax<close)
      ExtMax=close;

//--- predict next price
   PredictPrice();

   double heikenAshiOpen[], heikenAshiHigh[], heikenAshiLow[], heikenAshiClose[], heikenColor[];
   CopyBuffer(handle_iCustomHeiken, 0, 0, 2, heikenAshiOpen);
   CopyBuffer(handle_iCustomHeiken, 1, 0, 2, heikenAshiHigh);
   CopyBuffer(handle_iCustomHeiken, 2, 0, 2, heikenAshiLow);
   CopyBuffer(handle_iCustomHeiken, 3, 0, 2, heikenAshiClose);
   CopyBuffer(handle_iCustomHeiken, 4, 0, 2, heikenColor);

   double adx[];
   CopyBuffer(handle_adx, 0, 0, 2, adx);

   double volume[];
   CopyBuffer(handle_volume, 0, 0, 2, volume);

   Comment("heikenAshiOpen ", DoubleToString(heikenAshiOpen[0], _Digits),
           "\n heikenAshiHigh ", DoubleToString(heikenAshiHigh[0], _Digits),
           "\n heikenAshiLow ", DoubleToString(heikenAshiLow[0], _Digits),
           "\n heikenAshiClose ", DoubleToString(heikenAshiClose[0], _Digits),
           "\n Volume: ", DoubleToString(volume[0], _Digits));

   int currentBarsCount = Bars(_Symbol, _Period);

   if(currentBarsCount != lastBarsCount)
     {
      lastBarsCount = currentBarsCount;

      if((heikenColor[0] == 0.0 && heikenColor[1] == 1.0 && Order == -1 || Order == -1 && adx[0] < ADXThreshold) ||
         (heikenColor[0] == 1.0 && heikenColor[1] == 0.0 && Order == 1 || Order == 1 && adx[0] < ADXThreshold))
        {
         CloseOrders();
        }
      if(heikenColor[0] == 0.0 && heikenColor[1] == 1.0 && adx[0] > ADXThresholdOpen && volume[0] > VolumeThreshold && lastPredicted<predicted)
        {
         Order = 1;
         SignalLong();
        }
      if(heikenColor[0] == 1.0 && heikenColor[1] == 0.0 && adx[0] > ADXThresholdOpen && volume[0] > VolumeThreshold && lastPredicted>predicted)
        {
         Order = -1;
         SignalShort();
        }
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SignalLong()
  {
   if(PositionsTotal() == 0 && InpUseStops)
     {
      double lotSize = 0.1; // Tamaño del lote
      double pointSize = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
      double pipSize = pointSize * 10;

      // Abrir una orden de compra
      MqlTick mqltick;
      SymbolInfoTick(_Symbol, mqltick);
      double ask = mqltick.ask;

      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      int copied = CopyRates(_Symbol, 0, 0, 1, rates);
      double entryPrice = rates[0].close;

      Sl = NormalizeDouble(entryPrice - StopLossPips * pipSize, digits);
      Tp = NormalizeDouble(entryPrice + TakeProfitPips * pipSize, digits);

      int ticket = trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, InpLots, ask, Sl, Tp, "Buy order opened");

      if(ticket > 0)
         Print("Orden de compra abierta con éxito. Ticket: ", ticket);
      else
         Print("Error al abrir orden de compra. Código de error: ", GetLastError());
     }
   else
      if(PositionsTotal() == 0 && InpUseStops == false)
        {
         double price, sl = 0, tp = 0;
         MqlTick mqltick;
         SymbolInfoTick(_Symbol, mqltick);
         double ask = mqltick.ask;
         price = ask;
         trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, InpLots, price, sl, tp);
        }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SignalShort()
  {
   if(PositionsTotal() == 0 && InpUseStops)
     {
      double lotSize = 0.1; // Tamaño del lote
      double pointSize = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
      double pipSize = pointSize * 10;

      // Abrir una orden de venta
      MqlTick mqltick;
      SymbolInfoTick(_Symbol, mqltick);
      double bid = mqltick.bid;

      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      int copied = CopyRates(_Symbol, 0, 0, 1, rates);
      double entryPrice = rates[0].close;

      Sl = NormalizeDouble(entryPrice + StopLossPips * pipSize, digits);
      Tp = NormalizeDouble(entryPrice - TakeProfitPips * pipSize, digits);

      int ticket = trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, InpLots, bid, Sl, Tp, "Sell order opened");

      if(ticket > 0)
         Print("Orden de venta abierta con éxito. Ticket: ", ticket);
      else
         Print("Error al abrir orden de venta. Código de error: ", GetLastError());
     }
   else
      if(PositionsTotal() == 0 && InpUseStops == false)
        {
         double price, sl = 0, tp = 0;
         MqlTick mqltick;
         SymbolInfoTick(_Symbol, mqltick);
         double bid = mqltick.bid;
         price = bid;
         trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, InpLots, price, sl, tp);
        }
  }

//+------------------------------------------------------------------+
void CloseOrders()
  {
   if(PositionsTotal() != 0)
     {
      trade.PositionClose(_Symbol);
     }
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Get minimal and maximal Close for last 120 days                  |
//+------------------------------------------------------------------+
void GetMinMax(void)
  {
   vectorf close;
   close.CopyRates(_Symbol,PERIOD_D1,COPY_RATES_CLOSE,0,SAMPLE_SIZE);
   ExtMin=close.Min();
   ExtMax=close.Max();
  }
//+------------------------------------------------------------------+
//| Predict next price                                               |
//+------------------------------------------------------------------+
void PredictPrice(void)
  {
   static vectorf output_data(1);            // vector to get result
   static vectorf x_norm(SAMPLE_SIZE);       // vector for prices normalize

//--- check for normalization possibility
   if(ExtMin>=ExtMax)
     {
      Print("ExtMin>=ExtMax");
      ExtPredictedClass=-1;
      return;
     }
//--- request last bars
   if(!x_norm.CopyRates(_Symbol,PERIOD_D1,COPY_RATES_CLOSE,1,SAMPLE_SIZE))
     {
      Print("CopyRates ",x_norm.Size());
      ExtPredictedClass=-1;
      return;
     }
   float last_close=x_norm[SAMPLE_SIZE-1];
//--- normalize prices
   x_norm-=ExtMin;
   x_norm/=(ExtMax-ExtMin);
//--- run the inference
   if(!OnnxRun(ExtHandle,ONNX_NO_CONVERSION,x_norm,output_data))
     {
      Print("OnnxRun");
      ExtPredictedClass=-1;
      return;
     }
   lastPredicted=predicted;
//--- denormalize the price from the output value
   predicted=output_data[0]*(ExtMax-ExtMin)+ExtMin;
   Print("Predicted ",predicted, " last predicted: ", lastPredicted);
   //m_chart_h_line.Create(0,"Close",0,predicted);  // -delta00);  // set price to 0 to hide the line
   //m_chart_h_line.Color(clrDarkGoldenrod);
//--- classify predicted price movement

  }
//+------------------------------------------------------------------+
