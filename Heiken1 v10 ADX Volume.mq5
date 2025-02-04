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
input double InpLots = 0.01;            // Lotes para abrir posición
input bool InpUseStops = false;         // Use stops in trading
input int StopLossPips = 50;            // Stop Loss en pips
input int TakeProfitPips = 100;         // Take Profit en pips
input int VolumeThreshold = 4000;        // Umbral de volumen para confirmar la entrada
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

input int ADXThreshold = 20;            // ADX Close
input int ADXThresholdOpen = 20;        // ADX Open
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

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
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
      if(heikenColor[0] == 0.0 && heikenColor[1] == 1.0 && adx[0] > ADXThresholdOpen && volume[0] > VolumeThreshold)
        {
         Order = 1;
         SignalLong();
        }
      if(heikenColor[0] == 1.0 && heikenColor[1] == 0.0 && adx[0] > ADXThresholdOpen && volume[0] > VolumeThreshold)
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
