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
CTrade         trade;

// HA colors
color hacolor;

// Calculation HA Values
double haopen[];
double haclose[];
double hahigh[];
double halow[];

input ENUM_TIMEFRAMES my_timeframe_Heiken=PERIOD_CURRENT;
input ENUM_TIMEFRAMES my_timeframe_EMA=PERIOD_CURRENT;
input ENUM_TIMEFRAMES my_timeframe_adx=PERIOD_CURRENT;
input double InpLots = 0.01;     // Lotes para abrir posición
input bool   InpUseStops   = false;   // Use stops in trading
input int StopLossPips = 50;    // Stop Loss en Pips
input int TakeProfitPips = 100;  // Take Profit en Pips
input bool exponential = true; // Uso de media móvil exponencial
input int ADXThreshold = 20;  // ADX Close
input int ADXThresholdOpen = 20;  // ADX Open

int handle_iCustomHeiken;
int handle_ema100;
int handle_adx;
double Sl = 0.0;
double Tp = 0.0;
int lastBarsCount = 0;

double Close[];

int Order=0;


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
   handle_iCustomHeiken=iCustom(_Symbol,my_timeframe_Heiken,"\\Indicators\\Heiken_Ashi_copy");
   if(handle_iCustomHeiken==INVALID_HANDLE)
     {
      PrintFormat("Failed to create handle of the iCustom indicator for the symbol %s/%s, error code %d",
                  _Symbol,
                  EnumToString(my_timeframe_Heiken),
                  GetLastError());
      return(INIT_FAILED);
     }

   handle_ema100=iMA(_Symbol, my_timeframe_EMA, 100, 0, exponential ? MODE_EMA : MODE_SMA, PRICE_CLOSE);
   if(handle_ema100==INVALID_HANDLE)
     {
      PrintFormat("Failed to create handle of the iCustom indicator for the symbol %s/%s, error code %d",
                  _Symbol,
                  EnumToString(my_timeframe_EMA),
                  GetLastError());
      return(INIT_FAILED);
     }
   handle_adx=iADX(_Symbol, my_timeframe_adx, 14);
   if(handle_adx==INVALID_HANDLE)
     {
      PrintFormat("Failed to create handle of the iCustom indicator for the symbol %s/%s, error code %d",
                  _Symbol,
                  EnumToString(my_timeframe_adx),
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
   if(CopyBuffer(handle_iCustomHeiken, 0, 0, 2, heikenAshiOpen) <= 0 ||
      CopyBuffer(handle_iCustomHeiken, 1, 0, 2, heikenAshiHigh) <= 0 ||
      CopyBuffer(handle_iCustomHeiken, 2, 0, 2, heikenAshiLow) <= 0 ||
      CopyBuffer(handle_iCustomHeiken, 3, 0, 2, heikenAshiClose) <= 0 ||
      CopyBuffer(handle_iCustomHeiken, 4, 0, 2, heikenColor) <= 0)
     {
      Print("Error copying Heiken Ashi buffers");
      return;
     }

   double ema100[2];
   CopyBuffer(handle_ema100, 0, 0, 2, ema100);

   double adx[];
   CopyBuffer(handle_adx, 0, 0, 2, adx);

   Comment("Heiken Ashi Open: ", DoubleToString(heikenAshiOpen[0], _Digits),
           "\nHeiken Ashi High: ", DoubleToString(heikenAshiHigh[0], _Digits),
           "\nHeiken Ashi Low: ", DoubleToString(heikenAshiLow[0], _Digits),
           "\nHeiken Ashi Close: ", DoubleToString(heikenAshiClose[0], _Digits),
           "\nEMA100: ", DoubleToString(ema100[0], _Digits));

   int currentBarsCount = Bars(_Symbol, _Period);

   if(currentBarsCount != lastBarsCount)
     {
      lastBarsCount = currentBarsCount;

      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      int copied = CopyRates(_Symbol, 0, 0, 2, rates);
      if(copied <= 0)
        {
         Print("Error copying rates");
         return;
        }

      double currentClose = rates[0].close;
      double previousClose = rates[1].close;
      double heikenAshiSize = MathAbs(heikenAshiOpen[0] - heikenAshiClose[0]);
      Print("EMA ", ema100[0]);
      Print("HA_open",heikenAshiOpen[0]);
      Print("HA_open 1 ",heikenAshiOpen[1]);
      Print("HA_close ", heikenAshiClose[0]);
      Print("HA_close 1 ", heikenAshiClose[1]);

      if((currentClose < ema100[0]
         && Order==1) || (adx[0]<ADXThreshold && Order==1))
        {
         CloseOrders();
         Order=0;
        }
      if((currentClose > ema100[0]
         && Order==-1) || (adx[0]<ADXThreshold && Order==-1))
        {
         CloseOrders();
         Order=0;
        }

      if(currentClose > ema100[0] && heikenAshiSize < MathAbs(heikenAshiOpen[1] - heikenAshiClose[1]) && heikenAshiOpen[0] < heikenAshiClose[0] && adx[0]>ADXThresholdOpen)
        {
         Print("Buy");
         Order=1;
         SignalLong();
        }

      if(currentClose < ema100[0] && heikenAshiSize < MathAbs(heikenAshiOpen[1] - heikenAshiClose[1]) && heikenAshiOpen[0] > heikenAshiClose[0] && adx[0]>ADXThresholdOpen)
        {
         Print("Sell");
         Order=-1;
         SignalShort();
        }

     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SignalLong()
  {
   if(PositionsTotal() == 0)
     {
      double lotSize = InpLots;
      double pointSize = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
      double pipSize = pointSize * 10;

      MqlTick mqltick;
      SymbolInfoTick(_Symbol, mqltick);
      double ask = mqltick.ask;

      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      int copied = CopyRates(_Symbol, 0, 0, 1, rates);
      if(copied <= 0)
        {
         Print("Error copying rates");
         return;
        }

      double entryPrice = rates[0].close;

      if(InpUseStops)
        {
         Sl = NormalizeDouble(entryPrice - StopLossPips * pipSize, digits);
         Tp = NormalizeDouble(entryPrice + TakeProfitPips * pipSize, digits);
        }
      else
        {
         Sl = 0;
         Tp = 0;
        }

      int ticket = trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, lotSize, ask, Sl, Tp, "Buy order opened");
      if(ticket > 0)
         Print("Orden de compra abierta con éxito. Ticket: ", ticket);
      else
         Print("Error al abrir orden de compra. Código de error: ", GetLastError());
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SignalShort()
  {
   if(PositionsTotal() == 0)
     {
      double lotSize = InpLots;
      double pointSize = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
      double pipSize = pointSize * 10;

      MqlTick mqltick;
      SymbolInfoTick(_Symbol, mqltick);
      double bid = mqltick.bid;

      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      int copied = CopyRates(_Symbol, 0, 0, 1, rates);
      if(copied <= 0)
        {
         Print("Error copying rates");
         return;
        }

      double entryPrice = rates[0].close;

      if(InpUseStops)
        {
         Sl = NormalizeDouble(entryPrice + StopLossPips * pipSize, digits);
         Tp = NormalizeDouble(entryPrice - TakeProfitPips * pipSize, digits);
        }
      else
        {
         Sl = 0;
         Tp = 0;
        }

      int ticket = trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, lotSize, bid, Sl, Tp, "Sell order opened");
      if(ticket > 0)
         Print("Orden de venta abierta con éxito. Ticket: ", ticket);
      else
         Print("Error al abrir orden de venta. Código de error: ", GetLastError());
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CloseOrders()
  {
   while(PositionsTotal() > 0)
     {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         ulong ticket = PositionGetTicket(i);
         trade.PositionClose(ticket);
        }
     }
  }

//+------------------------------------------------------------------+
