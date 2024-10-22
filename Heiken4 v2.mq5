//+------------------------------------------------------------------+
//|                                                      ProjectName |
//|                                      Copyright 2020, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
CTrade trade;

// Input parameters
input ENUM_TIMEFRAMES maTimeframe = PERIOD_H1;
input ENUM_TIMEFRAMES adxTimeframe = PERIOD_H1;
input int maLength = 55;
input ENUM_APPLIED_PRICE maPrice = PRICE_CLOSE;
input int adxPeriod = 14;
input int adxThreshold = 25;
input double lotSize = 0.01;


int handleMa;
int handleAdx;
double lastRenkoClose = 0.0;
int Order = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
// Create handles for indicators
   handleMa = iMA(_Symbol, maTimeframe, maLength, 0, MODE_EMA, maPrice);
   if(handleMa == INVALID_HANDLE)
     {
      Print("Error creating handle for MA");
      return INIT_FAILED;
     }

   handleAdx = iADX(_Symbol, adxTimeframe, adxPeriod);
   if(handleAdx == INVALID_HANDLE)
     {
      Print("Error creating handle for ADX");
      return INIT_FAILED;
     }

   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
// Release handles
   IndicatorRelease(handleMa);
   IndicatorRelease(handleAdx);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   double maBuffer[2];
   double adxBuffer[1];

   if(CopyBuffer(handleMa, 0, 0, 2, maBuffer) <= 0 ||
      CopyBuffer(handleAdx, 0, 0, 1, adxBuffer) <= 0)
     {
      Print("Error copying indicator buffers");
      return;
     }

   bool maRising = maBuffer[0] > maBuffer[1];
   bool maFalling = maBuffer[0] < maBuffer[1];
   bool adxAboveThreshold = adxBuffer[0] > adxThreshold;

   double renkoClose = lastRenkoClose + (maRising ? 1 : (maFalling ? -1 : 0));

   bool buySignal = maRising && adxAboveThreshold;
   bool sellSignal = maFalling && adxAboveThreshold;

// Check for existing positions
   bool longPosition = false;
   bool shortPosition = false;

   /*  for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
         if(PositionSelect(PositionGetTicket(i)))
         {
             longPosition |= (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
             shortPosition |= (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL);
         }
     }  */



   if(sellSignal && Order == 1 && PositionsTotal()>0)
     {
      CloseLongPositions();
      Order = 0; // Reset order variable
     }

   if(buySignal && Order == -1 && PositionsTotal()>0)
     {
      CloseShortPositions();
      Order = 0; // Reset order variable
     }

   if(buySignal &&  PositionsTotal()==0)
     {
      if(trade.Buy(lotSize, _Symbol))
        {
         Order = 1;
        }
      else
        {
         Print("Error opening buy order: ", GetLastError());
        }
     }

   if(sellSignal &&  PositionsTotal()==0)
     {
      if(trade.Sell(lotSize, _Symbol))
        {
         Order = -1;
        }
      else
        {
         Print("Error opening sell order: ", GetLastError());
        }
     }


   lastRenkoClose = renkoClose;
  }

//+------------------------------------------------------------------+
//| Close all long positions                                         |
//+------------------------------------------------------------------+
void CloseLongPositions()
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
//| Close all short positions                                        |
//+------------------------------------------------------------------+
void CloseShortPositions()
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
