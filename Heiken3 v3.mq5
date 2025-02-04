//+------------------------------------------------------------------+
//|                                              NemethStrategy.mq5  |
//|              Copyright 2024, Javier S. Gastón de Iriarte Cabrera |
//|                      https://www.mql5.com/en/users/jsgaston/news |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Javier S. Gastón de Iriarte Cabrera"
#property link      "https://www.mql5.com/en/users/jsgaston/news"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
CTrade trade;

// Input parameters
input ENUM_TIMEFRAMES middle_timeframe = PERIOD_H1;
input ENUM_TIMEFRAMES higher_timeframe = PERIOD_D1;
input double renkoSize = 1.0;
input int emaPeriod = 20;
input double lotSize = 0.01;

int handle_iCustomHeikenMiddle;
int handle_iCustomHeikenHigher;
int handle_emaMiddle;

double lastRenkoClose = 0.0;

int Order=0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   handle_iCustomHeikenMiddle = iCustom(_Symbol, middle_timeframe, "\\Indicators\\Heiken_Ashi_copy");
   if(handle_iCustomHeikenMiddle == INVALID_HANDLE)
     {
      Print("Error creating handle for Heiken Ashi Middle Timeframe");
      return INIT_FAILED;
     }

   handle_iCustomHeikenHigher = iCustom(_Symbol, higher_timeframe, "\\Indicators\\Heiken_Ashi_copy");
   if(handle_iCustomHeikenHigher == INVALID_HANDLE)
     {
      Print("Error creating handle for Heiken Ashi Higher Timeframe");
      return INIT_FAILED;
     }

   handle_emaMiddle = iMA(_Symbol, middle_timeframe, emaPeriod, 0, MODE_EMA, PRICE_CLOSE);
   if(handle_emaMiddle == INVALID_HANDLE)
     {
      Print("Error creating handle for EMA");
      return INIT_FAILED;
     }

   return INIT_SUCCEEDED;
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
// Cleanup code here
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   double heikenAshiMiddleOpen[], heikenAshiMiddleClose[];
   double heikenAshiHigherOpen[], heikenAshiHigherClose[];
   double emaMiddle[];

   if(CopyBuffer(handle_iCustomHeikenMiddle, 0, 0, 2, heikenAshiMiddleOpen) <= 0 ||
      CopyBuffer(handle_iCustomHeikenMiddle, 3, 0, 2, heikenAshiMiddleClose) <= 0 ||
      CopyBuffer(handle_iCustomHeikenHigher, 0, 0, 2, heikenAshiHigherOpen) <= 0 ||
      CopyBuffer(handle_iCustomHeikenHigher, 3, 0, 2, heikenAshiHigherClose) <= 0 ||
      CopyBuffer(handle_emaMiddle, 0, 0, 2, emaMiddle) <= 0)
     {
      Print("Error copying indicator buffers");
      return;
     }

   bool greenMiddle = heikenAshiMiddleClose[0] > heikenAshiMiddleOpen[0];
   bool redMiddle = heikenAshiMiddleClose[0] < heikenAshiMiddleOpen[0];
   bool greenHigher = heikenAshiHigherClose[0] > heikenAshiHigherOpen[0];
   bool redHigher = heikenAshiHigherClose[0] < heikenAshiHigherOpen[0];

   bool slopeUpMiddle = emaMiddle[0] > emaMiddle[1];
   bool slopeDownMiddle = emaMiddle[0] < emaMiddle[1];

   double renkoClose = lastRenkoClose + (greenMiddle ? renkoSize : (redMiddle ? -renkoSize : 0));

   bool renkoGreen = renkoClose > lastRenkoClose;
   bool renkoRed = renkoClose < lastRenkoClose;

   bool buy = greenMiddle && greenHigher && slopeUpMiddle && renkoGreen;
   bool sell = redMiddle || redHigher || slopeDownMiddle || renkoRed;
   bool shortSell = redMiddle && redHigher && slopeDownMiddle && renkoRed;
   bool cover = greenMiddle || greenHigher || slopeUpMiddle || renkoGreen;

// Check for existing positions
   bool longPosition = false;
   bool shortPosition = false;
   /*for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionSelect(PositionGetTicket(i)))
        {
         longPosition |= (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
         shortPosition |= (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL);
        }
     }*/
   if(sell && Order==1 && PositionsTotal()>0)
     {
      CloseLongPositions();
     }

   if(cover && Order==-1 && PositionsTotal()>0)
     {
      CloseShortPositions();
     }

   if(buy && Order==-1 && PositionsTotal()==0)
     {
      CloseShortPositions();
      trade.Buy(lotSize, _Symbol, 0, 0, 0, "Buy Order");
      Order=1;
     }

   if(shortSell && Order==1 && PositionsTotal()==0)
     {
      CloseLongPositions();
      trade.Sell(lotSize, _Symbol, 0, 0, 0, "Sell Order");
      Order=-1;
     }
   if(buy && Order==0 && PositionsTotal()==0)
     {
      //CloseShortPositions();
      trade.Buy(lotSize, _Symbol, 0, 0, 0, "Buy Order");
      Order=1;
     }

   if(shortSell && Order==0 && PositionsTotal()==0)
     {
      //CloseLongPositions();
      trade.Sell(lotSize, _Symbol, 0, 0, 0, "Sell Order");
      Order=-1;
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
