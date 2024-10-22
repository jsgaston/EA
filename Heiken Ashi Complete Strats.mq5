//+------------------------------------------------------------------+
//|                         Heiken Ashi Trend Following Strategy.mq5 |
//|              Copyright 2024, Javier S. Gastón de Iriarte Cabrera |
//|                      https://www.mql5.com/en/users/jsgaston/news |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Javier S. Gastón de Iriarte Cabrera"
#property link      "https://www.mql5.com/en/users/jsgaston/news"
#property version   "1.01"

#include <Trade\Trade.mqh> //Instatiate Trades Execution Library
#include <Trade\OrderInfo.mqh> //Instatiate Library for Orders Information
#include <Trade\PositionInfo.mqh> // Library for all position features and information

CTrade trade;

#define SAMPLE_SIZE 120

#resource "/Files/model.eurusd.D1.120.till2023.onnx" as uchar ExtModel[]
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
//--- input parameters
int MA_Period = 14;

input group                "---- Type of Strategy ----"

enum ENUM_Strat_TYPE
  {
   SimpleStrat   = 0,  // Oscilators & Indicators
   PredictStrat  = 1,  // With Predictions
  };
input ENUM_Strat_TYPE        inp_strat_type               = SimpleStrat;   // Select type of Strategy



input group                "---- Strategy ----"
enum ENUM_Strategy_TYPE
  {
   trend  = 0, // Trend Heiken based
   CCI  = 1,   // CCI Heiken based
   Volume  = 2,// Volume Heiken based
   S_P  = 3,   // Support & Resistance Levels Heiken based
   Fibo  = 4,  // Fibonacci Retracement Heiken based
   MAs  = 5,   // 2 MAs crossover Heiken based
   RSI = 6,    // RSI Heiken based
   Bol = 7,    // Bollinger Bands Heiken based
   MACD = 8,   // MACD Heiken based
   Stoch = 9,  // Stochastic Heiken based
   PSAR = 10,  // PSAR Heiken based
   REN = 11,   // Renko Heiken based
   RENB = 12,  // Renko plan B Heiken based
   WIS = 13,   // Wise MAs based



  };
input ENUM_Strategy_TYPE        inp_strategy_type               = trend;   // Select type of Strategy

input group                "---- Lots ----"

enum ENUM_LOT_TYPE
  {
   LOT_TYPE_FIX   = 0,  // fix lot
   LOT_TYPE_RISK  = 1,  // risk %
  };

input ENUM_LOT_TYPE        inp_lot_type               = LOT_TYPE_RISK;              // type of lot
input double               inp_lot_fix                = 0.1;                        // fix lot
input double               inp_lot_risk               = 2.0;                        // margin %
//input double InpLots = 0.10;     // Lots open position


input group                "---- Stops & Filters ----"
enum ENUM_Stops_TYPE
  {
   ATR  = 0,                       // ATR
   PIPs = 1,                       // PIPs
   PIP_Vol  = 2,                   // PIPs + Volume
   ATR_Vol = 3,                    // ATR + Volume
   ADX_Pip   = 4,                  // ADX + PIPs
   ADX_Atr   = 5,                  // ADX + ATR
   ADX_Pip_Vol   = 6,              // ADX + PIPs + Volume
   ADX_Atr_Vol   = 7,              // ADX + ATR + Volume
   WO   = 8,                       // With Out Stops
  };
input ENUM_Stops_TYPE inp_stops_type = PIPs;   // Select type of Stops

input int StopLossPips = 300;      // Stop Loss in Pips
input int TakeProfitPips = 600;    // Take Profit in Pips
//*input int ADXThreshold = 20;    // Threshold for ADX to consider a strong trend
//input bool StopsATR = true;      // Stops by ATR
input group                "---- ATR ----"
input int ATRPeriod = 14;          // ATR Period
input double atr_multi = 1.5;      // ATR Multiplier

input group                "---- ADX ----"
input int ADX_Period = 14;                   // ADX Period
input double ADX_threshold_over = 20.0;      // ADX Over Threshold
input double ADX_threshold_under = 45.0;     // ADX Under Threshold
input ENUM_TIMEFRAMES ADX_Timeframe=PERIOD_CURRENT;//ADX Timeframe

input group                "---- MAs ----"
//--- input parameters
input int Short_Period = 14;                // MA Short Period & Trend Period
input ENUM_TIMEFRAMES MA_Timeframe=PERIOD_CURRENT;//MA short Timeframe
input int Long_Period = 28;                 // MA Long Period
input ENUM_TIMEFRAMES MA2_Timeframe=PERIOD_CURRENT;//MA long Timeframe

input group                "---- Trading Hours ----"
enum ENUM_Days_TYPE
  {
   Sun   = 0,                  // Sunday
   Mon   = 1,                  // Monday
   Tue   = 2,                  // Tuesday
   Wed   = 3,                  // Wednesday
   Thu   = 4,                  // Thirsday
   Fri   = 5,                  // Friday
   Sat   = 6,                  // Saturday


  };
input bool market_hours = false;           // Trade hours or days?
input int before = 1;                      // hours from where before not do trading
input int after = 23;                      // hours till where after not do trading
input ENUM_Days_TYPE Saturday = Sat;       // From day (for example: Saturday) don't trade
input ENUM_Days_TYPE Sunday = Sun;         // To day (for example: Sunday) (do not trade)

input group                "---- Volume ----"
input bool Volume_Filter = true;          // Volume filter?
input int VolumeThreshold = 750;          // Volume Threshold
input int vol_shift = 0;                   // Volume Shift
int VOLVOL = 0;
double volumen_barra;

input group                "---- CCI ----"
input int CCI_period = 14;                 // Period
input double CCI_Buy_Level = -100;         // Buy Level
input double CCI_Sell_Level = 100;         // Sell Level
input ENUM_TIMEFRAMES CCI_Timeframe=PERIOD_CURRENT;//CCI Timeframe


input group                "---- Support & Resistance ----"
input double Support_Level = 1.08000;      //Support Level
input double Resistance_Level = 1.07650;   // Resistance Level

input group                "---- RSI ----"
//--- input parameters
input int RSI_Period = 14;                 // RSI Period
input double Overbought = 70;              // Overbought
input int Oversold = 30;                   // Oversold
input ENUM_TIMEFRAMES RSI_Timeframe=PERIOD_CURRENT;//RSI Timeframe

input group                "---- Bollinger Bands ----"
//--- input parameters
input int Bands_Period = 20;               // Bollinger Bands
input int bands_shift = 0;                 // Bollinger Bands shift
input ENUM_TIMEFRAMES BOL_Timeframe=PERIOD_CURRENT;//Bollinger Timeframe

input group                "---- MACD ----"
//--- input parameters
input int FastEMA = 12;                   // Fast EMA
input int SlowEMA = 26;                   // Slow EMA
input int SignalSMA = 9;                  // Sigma EMA
input ENUM_TIMEFRAMES MACD_Timeframe=PERIOD_CURRENT;//MACD Timeframe

input group                "---- Stochastic ----"
//--- input parameters
input int K_Period = 14;                 //K Period
input int D_Period = 3;                  //D Period
input int Slowing = 3;                   //Slowing
input ENUM_TIMEFRAMES STO_Timeframe=PERIOD_CURRENT;//Stochcastic Timeframe

input group                "---- Parbolic SAR ----"
//--- input parameters
input double Step = 0.02;                 //Step
input double Maximum = 0.2;               //Maximum
input ENUM_TIMEFRAMES PSA_Timeframe=PERIOD_CURRENT;//PSAR Timeframe

input group                "---- RENKO ----"
//--- input parameters
input string myTextInput1 = "Short Period";       // MA Time Frame
input string myTextInput2 = "Short Period";       // MA Perdiod
input string myTextInput3 = "ADX Time Frame";     // ADX Time Frame
input string myTextInput4 = "ADX Period";         // ADX Period
input string myTextInput5 = "ADX Threshold";      // ADX Threshold
double lastRenkoClose = 100.0;                    // last Renko

input group                "---- Heiken Ashi ----"
//--- input parameters
input ENUM_TIMEFRAMES HEI_Timeframe=PERIOD_CURRENT;//Heiken Ashi Timeframe

input group                "---- Wise ----"
//--- input parameters
input string myTextInput6 = "MAs";         // MAs Fast & Slow

//--- global variables
double VWMA[];
double         parSAR[];

int MA_Handle;
int MA_Handle2;
int Volume_Handle;
int CCI_Handle;
int RSI_Handle;
int MACD_Handle;
int Stochastic_Handle;
int Parabolic_SAR_Handle;

double Sl = 0.0;
double Tp = 0.0;

enum typets {Simple,MoralExp,None};
input group                "---- Trailing Stop ----"
input ENUM_TIMEFRAMES TFTS=PERIOD_M1;//Trailing stop timeframe
input ushort Seed=0;
input uchar Slippage=10;
input typets TypeTS=Simple;
input bool UseTakeProfit=true,
           MultiTS=false;

int up[],dn[];//arrays for storing statistics
int buy_sl,buy_tp,sell_sl,sell_tp;
double pointvalue;//point price
double SLBuy=0,SLSell=0,SLNeutral=0;

int ATRHandle;
int ADX_Handle;

//--- global variables
double VolumeOscillator[];
int handle_iCustomHeiken;
int handle_iCustomFibo;
int Bands_Handle;
int Order=0;
double         UpperBuffer[];
double         LowerBuffer[];
double         MiddleBuffer[];

double         kvalue[];
double         dvalue[];

long     ExtHandle=INVALID_HANDLE;
int      ExtPredictedClass=-1;
datetime ExtNextBar=0;
datetime ExtNextDay=0;
float    ExtMin=0.0;
float    ExtMax=0.0;

double lastPredicted = 0.0;
double predicted = 0.0;

int variable_inutil=0;

double ma00, ma01, ma10, ma11 =0.0;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {

   ADX_Handle = iADX(Symbol(), ADX_Timeframe, ADX_Period);
   if(ADX_Handle == INVALID_HANDLE)
     {
      Print("Error initializing ADX indicator: ", GetLastError());
      return INIT_FAILED;
     }

   MA_Handle = iMA(Symbol(), MA_Timeframe, Short_Period,0,MODE_SMA,PRICE_CLOSE);
   if(MA_Handle == INVALID_HANDLE)
     {
      Print("Error initializing MA indicator: ", GetLastError());
      return INIT_FAILED;
     }

   Volume_Handle = iVolumes(_Symbol, PERIOD_CURRENT,VOLUME_TICK);
   if(Volume_Handle == INVALID_HANDLE)
     {
      Print("Error initializing Volume indicator: ", GetLastError());
      return INIT_FAILED;
     }


   MA_Handle2 = iMA(_Symbol, MA2_Timeframe, Long_Period,0,MODE_SMA,PRICE_CLOSE);
   if(MA_Handle2 == INVALID_HANDLE)
     {
      Print("Error initializing MA2 indicator: ", GetLastError());
      return INIT_FAILED;
     }


   handle_iCustomHeiken=iCustom(_Symbol,HEI_Timeframe,"\\Indicators\\Heiken_Ashi_copy");
   if(handle_iCustomHeiken==INVALID_HANDLE)
     {
      //--- tell about the failure and output the error code
      PrintFormat("Failed to create handle of the iCustom indicator for the symbol %s/%s, error code %d",
                  _Symbol,
                  EnumToString(Period()),
                  GetLastError());
      //--- the indicator is stopped early
      return(INIT_FAILED);
     }



   CCI_Handle = iCCI(Symbol(), CCI_Timeframe,CCI_period,PRICE_CLOSE);
   if(CCI_Handle == INVALID_HANDLE)
     {
      Print("Error initializing CCI indicator: ", GetLastError());
      return INIT_FAILED;
     }



   handle_iCustomFibo=iCustom(_Symbol,Period(),"\\Indicators\\fibos");
   if(handle_iCustomFibo==INVALID_HANDLE)
     {
      //--- tell about the failure and output the error code
      PrintFormat("Failed to create handle of the iCustom indicator for the symbol %s/%s, error code %d",
                  _Symbol,
                  EnumToString(Period()),
                  GetLastError());
      //--- the indicator is stopped early
      return(INIT_FAILED);
     }


   RSI_Handle = iRSI(Symbol(), RSI_Timeframe, RSI_Period,PRICE_CLOSE);
   if(RSI_Handle == INVALID_HANDLE)
     {
      Print("Error initializing RSI indicator: ", GetLastError());
      return INIT_FAILED;
     }

   Bands_Handle = iBands(Symbol(), BOL_Timeframe, Bands_Period,0,2.0,PRICE_CLOSE);
   if(Bands_Handle == INVALID_HANDLE)
     {
      Print("Error initializing RSI indicator: ", GetLastError());
      return INIT_FAILED;
     }


   MACD_Handle = iMACD(Symbol(), MACD_Timeframe,FastEMA,SlowEMA,SignalSMA,PRICE_CLOSE);
   if(MACD_Handle == INVALID_HANDLE)
     {
      Print("Error initializing MACD indicator: ", GetLastError());
      return INIT_FAILED;
     }

   Stochastic_Handle = iStochastic(Symbol(), STO_Timeframe,K_Period,D_Period,Slowing,MODE_EMA,STO_LOWHIGH);
   if(Stochastic_Handle == INVALID_HANDLE)
     {
      Print("Error initializing Stochastic indicator: ", GetLastError());
      return INIT_FAILED;
     }

   Parabolic_SAR_Handle = iSAR(Symbol(), PSA_Timeframe,Step,Maximum);
   if(Parabolic_SAR_Handle == INVALID_HANDLE)
     {
      Print("Error initializing Parabolic SAR indicator: ", GetLastError());
      return INIT_FAILED;
     }




   /*ArrayResize(VWMA, 2);
   ArraySetAsSeries(VWMA, true);
   SetIndexBuffer(0, VWMA);
   ArrayResize(VolumeOscillator, 2);
   ArraySetAsSeries(VolumeOscillator, true);
   SetIndexBuffer(0, VolumeOscillator);*/

   if(Seed>0)//initialize random number generator
      MathSrand(Seed);
   else
      MathSrand(GetTickCount());

   trade.SetDeviationInPoints(Slippage);

   pointvalue=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE)*SymbolInfoDouble(_Symbol,SYMBOL_POINT)/SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);

   for(int i=iBars(_Symbol,TFTS)-1; i>0; i--)
     {
      double open=iOpen(_Symbol,TFTS,i);
      CalcLvl(up,(int)MathRound((iHigh(_Symbol,TFTS,i)-open)/_Point));
      CalcLvl(dn,(int)MathRound((open-iLow(_Symbol,TFTS,i))/_Point));
     }


   ATRHandle = iATR(Symbol(), Period(), ATRPeriod);
   if(ATRHandle == INVALID_HANDLE)
     {
      Print("Error initializing ATR indicator: ", GetLastError());
      return INIT_FAILED;
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
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- check new day

//Print(Order);


   double heikenAshiOpen[], heikenAshiHigh[], heikenAshiLow[], heikenAshiClose[], heikenColor[];
   CopyBuffer(handle_iCustomHeiken,0,0,2,heikenAshiOpen);
   CopyBuffer(handle_iCustomHeiken,1,0,2,heikenAshiHigh);
   CopyBuffer(handle_iCustomHeiken,2,0,2,heikenAshiLow);
   CopyBuffer(handle_iCustomHeiken,3,0,2,heikenAshiClose);
   CopyBuffer(handle_iCustomHeiken,4,0,2,heikenColor);
   Comment("heikenAshiOpen ",DoubleToString(heikenAshiOpen[0],_Digits),
           "\n heikenAshiHigh ",DoubleToString(heikenAshiHigh[0],_Digits),
           "\n heikenAshiLow ",DoubleToString(heikenAshiLow[0],_Digits),
           "\n heikenAshiClose ",DoubleToString(heikenAshiClose[0],_Digits));

   double fibo_l0[],fibo_l1[],fibo_l2[],fibo_l3[],fibo_l4[],fibo_l5[],fibo_l6[],fibo_l7[];
   CopyBuffer(handle_iCustomFibo,0,0,2,fibo_l0);
   CopyBuffer(handle_iCustomFibo,1,0,2,fibo_l1);
   CopyBuffer(handle_iCustomFibo,2,0,2,fibo_l2);
   CopyBuffer(handle_iCustomFibo,3,0,2,fibo_l3);
   CopyBuffer(handle_iCustomFibo,4,0,2,fibo_l4);
   CopyBuffer(handle_iCustomFibo,5,0,2,fibo_l5);
   CopyBuffer(handle_iCustomFibo,6,0,2,fibo_l6);
   CopyBuffer(handle_iCustomFibo,7,0,2,fibo_l7);



   if(market_hours==true)
     {

      if(IsMarketClosed())  // Verificar si el mercado está cerrado
        {
         return; // Si el mercado está cerrado, no hacer nada
        }
     }
   double Volume[];
   double ma[];
   double ma2[];
   double atr[1];
   double cci[];
   double adx[];
   double rsi[];
   double macdMain[], macdSignal[];

   ArraySetAsSeries(Volume, true);
   ArraySetAsSeries(ma, true);
   ArraySetAsSeries(ma2, true);
   ArraySetAsSeries(adx, true);
   ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(UpperBuffer, true);
   ArraySetAsSeries(LowerBuffer, true);
   ArraySetAsSeries(MiddleBuffer, true);
   ArraySetAsSeries(macdMain, true);
   ArraySetAsSeries(macdSignal, true);
   ArraySetAsSeries(kvalue, true);
   ArraySetAsSeries(dvalue, true);
   ArraySetAsSeries(parSAR, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr);
   CopyBuffer(MA_Handle, 0, 0, 2, ma);
   CopyBuffer(MA_Handle2, 0, 0, 2, ma2);
   CopyBuffer(CCI_Handle, 0, 0, 2, cci);
   CopyBuffer(ADX_Handle, 0, 0, 2, adx);
   CopyBuffer(RSI_Handle, 0, 0, 2, rsi);
   CopyBuffer(Bands_Handle, 0, 0, 2, UpperBuffer);
   CopyBuffer(Bands_Handle, 1, 0, 2, LowerBuffer);
   CopyBuffer(Bands_Handle, 2, 0, 2, MiddleBuffer);
   CopyBuffer(MACD_Handle, 0, 0, 2, macdMain);
   CopyBuffer(MACD_Handle, 1, 0, 2, macdSignal);
   CopyBuffer(Stochastic_Handle, 0, 0, 2, kvalue);
   CopyBuffer(Stochastic_Handle, 1, 0, 2, dvalue);
   CopyBuffer(Parabolic_SAR_Handle, 0, 0, 2, parSAR);


   double Fast = ma[1];
   double Slow = ma2[1];

   if(CopyBuffer(ADX_Handle, 0, 0, ADX_Period, adx) <= 0)
     {
      Print("Error copying ADX buffer: ", GetLastError());
      return;
     }


   if(CopyBuffer(MA_Handle, 0, 0, Short_Period, ma) <= 0)
     {
      Print("Error copying MA buffer: ", GetLastError());
      return;
     }

   if(CopyBuffer(MA_Handle2, 0, 0, Long_Period, ma2) <= 0)
     {
      Print("Error copying MA buffer: ", GetLastError());
      return;
     }

   if(CopyBuffer(Volume_Handle, 0, 0, 1, Volume) <= 0)
     {
      Print("Error copying Volume buffer: ", GetLastError());
      return;
     }


   if(CopyBuffer(CCI_Handle, 0, 0, CCI_period, cci) <= 0)
     {
      Print("Error copying Bands buffer: ", GetLastError());
      return;
     }


   if(CopyBuffer(RSI_Handle, 0, 0, RSI_Period, rsi) <= 0)
     {
      Print("Error copying MA buffer: ", GetLastError());
      return;
     }


   if(CopyBuffer(MACD_Handle, 0, 0, Period(), macdMain) <= 0)
     {
      Print("Error copying MACD MAin buffer: ", GetLastError());
      return;
     }


   if(CopyBuffer(MACD_Handle, 1, 0, Period(), macdSignal) <= 0)
     {
      Print("Error copying MACD Signal buffer: ", GetLastError());
      return;
     }


   if(CopyBuffer(Stochastic_Handle, 0, 0, K_Period, kvalue) <= 0)
     {
      Print("Error copying Bands buffer: ", GetLastError());
      return;
     }


   if(CopyBuffer(Stochastic_Handle, 1, 0, D_Period, dvalue) <= 0)
     {
      Print("Error copying Bands buffer: ", GetLastError());
      return;
     }

   if(CopyBuffer(Parabolic_SAR_Handle, 0, 0,Period(), parSAR) <= 0)
     {
      Print("Error copying Parabolic_SAR buffer: ", GetLastError());
      return;
     }


   MqlRates rates[];
   ArraySetAsSeries(rates,true);
   int copied = CopyRates(_Symbol,0,0,MA_Period,rates);
   if(copied <= 0)
     {
      Print("Error copying rates: ", GetLastError());
      return;
     }

   if(CopyBuffer(Bands_Handle, 0, 0, Bands_Period, UpperBuffer) <= 0)
     {
      Print("Error copying Bands buffer: ", GetLastError());
      return;
     }


   if(CopyBuffer(Bands_Handle, 1, 0, Bands_Period, LowerBuffer) <= 0)
     {
      Print("Error copying Bands buffer: ", GetLastError());
      return;
     }

   if(CopyBuffer(Bands_Handle, 2, 0, Bands_Period, MiddleBuffer) <= 0)
     {
      Print("Error copying Bands buffer: ", GetLastError());
      return;
     }

   double entryPrice = rates[0].close;

   double heikCol = heikenColor[0];

   double Volumen = Volume[0];

   double upperBand = UpperBuffer[0];
   double LowerBand = LowerBuffer[0];
   double middleBand = MiddleBuffer[0];

   double MacdMain = macdMain[0];
   double MacdSignal = macdSignal[0];

   double KValue = kvalue[0];
   double DValue = dvalue[0];


   /*
      double VolumeOscillator_first = ma[0] - ma2[0];
      double VolumeOscillator_second = ma[1] - ma2[1];*/

   double pointSize = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double pipSize = pointSize * 10;

   double Ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double Bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   double atr_val = atr[0];


// Índice de la barra que queremos guardar (0 es la barra actual)
   int indice_barra = 1; // Por ejemplo, la barra anterior a la actual

// Obtener el valor del volumen de la barra específica
   volumen_barra = iVolume(Symbol(), PERIOD_CURRENT, indice_barra);

// Imprimir el valor del volumen en la ventana del diario
//Print("Valor de volumen de la barra ", indice_barra, ": ", volumen_barra);

   Volumen = volumen_barra;

   bool maRising = ma[0] > ma[1];
   bool maFalling = ma[0] < ma[1];

/////////////////////////////////////////////////////////////////

     {
      if(inp_strategy_type==0)
         Trend(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0]);
      if(inp_strategy_type==1)
         CCI(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, cci[0], adx[0]);
      if(inp_strategy_type==2)
        {
         Vol_strat(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0]);
        }
      if(inp_strategy_type==3)
         S_P(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], heikenAshiClose[0]);
      if(inp_strategy_type==4)
         Fibo(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, fibo_l4[0], adx[0], heikenAshiClose[0]);
      if(inp_strategy_type==5)
        {
         MAs(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, Fast, Slow, adx[0]);
        }
      if(inp_strategy_type==6)
         RSI(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], rsi[0]);
      if(inp_strategy_type==7)
         Bol(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], heikenAshiClose[0], middleBand);
      if(inp_strategy_type==8)
         MACD(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], MacdMain, MacdSignal);
      if(inp_strategy_type==9)
         Stoch(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], KValue, DValue);
      if(inp_strategy_type==10)
         PSAR(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], heikenAshiClose[0], parSAR[0]);
      if(inp_strategy_type==11)
         REN(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], maRising, maFalling);
      if(inp_strategy_type==12)
         RENB(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], maRising, maFalling);
      if(inp_strategy_type==13)
         WIS(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], ma[0], ma[1], ma2[0], ma2[1]);
     }






     {
      if(inp_strategy_type==0)
         Trend2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0]);
      if(inp_strategy_type==1)
         CCI2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, cci[0], adx[0]);
      if(inp_strategy_type==2)
        {
         Vol_strat2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0]);
        }
      if(inp_strategy_type==3)
         S_P2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], heikenAshiClose[0]);
      if(inp_strategy_type==4)
         Fibo2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, fibo_l4[0], adx[0], heikenAshiClose[0]);
      if(inp_strategy_type==5)
         MAs2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, Fast, Slow, adx[0]);
      if(inp_strategy_type==6)
         RSI2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], rsi[0]);
      if(inp_strategy_type==7)
         Bol2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], heikenAshiClose[0], middleBand);
      if(inp_strategy_type==8)
         MACD2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], MacdMain, MacdSignal);
      if(inp_strategy_type==9)
         Stoch2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], KValue, DValue);
      if(inp_strategy_type==10)
         PSAR2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], heikenAshiClose[0], parSAR[0]);
      if(inp_strategy_type==11)
         REN2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], maRising, maFalling);
      if(inp_strategy_type==12)
         RENB2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], maRising, maFalling);
      if(inp_strategy_type==13)
         WIS2(entryPrice, heikCol, Volumen, Ask, Bid, atr_val, digits, Order, adx[0], ma[0], ma[1], ma2[0], ma2[1]);
     }





//if(NewBar()==true)//open positions for the current timeframe

   if(NewBarTS()==true)//gather statistics and launch trailing stop
     {
      double open=iOpen(_Symbol,TFTS,1);
      CalcLvl(up,(int)MathRound((iHigh(_Symbol,TFTS,1)-open)/_Point));
      CalcLvl(dn,(int)MathRound((open-iLow(_Symbol,TFTS,1))/_Point));
      buy_sl=CalcSL(dn);
      buy_tp=CalcTP(up);
      sell_sl=CalcSL(up);
      sell_tp=CalcTP(dn);

      if(TypeTS==Simple)//simple trailing stop
         SimpleTS();

      if(TypeTS==MoralExp)//Moral expectation
         METS();
     }

   double bid=SymbolInfoDouble(_Symbol,SYMBOL_BID);

   if(bid==SLNeutral || bid<=SLBuy || (SLSell>0 && bid>=SLSell))
     {
      for(int i=PositionsTotal()-1; i>=0; i--)
        {
         ulong ticket=PositionGetTicket(i);
         if(PositionSelectByTicket(ticket)==true)
            trade.PositionClose(ticket);
        }
     }

////////////////////////////////////////////////////////////////


  }
//+------------------------------------------------------------------+
void OnTrade()
  {
//---
   if(MultiTS==true)//common trailing stop
      AllTS();
//---
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void AllTS()
  {
//---
   double lot_buy=0,lot_sell=0,ask_opt=0,bid_opt=0,spread=SymbolInfoInteger(_Symbol,SYMBOL_SPREAD)*_Point;
   SLNeutral=0;
   SLBuy=0;
   SLSell=0;

   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      ulong ticket=PositionGetTicket(i);
      if(PositionSelectByTicket(ticket)==true)
        {
         double swap=PositionGetDouble(POSITION_SWAP)-PositionGetDouble(POSITION_COMMISSION),
                lot=PositionGetDouble(POSITION_VOLUME);

         if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
           {
            double price=PositionGetDouble(POSITION_PRICE_OPEN)-swap/(lot*pointvalue);
            lot_buy=lot_buy+lot;
            ask_opt=ask_opt+(price+spread)*lot;
            bid_opt=bid_opt+price*lot;
           }
         else
           {
            double price=PositionGetDouble(POSITION_PRICE_OPEN)+swap/(lot*pointvalue);
            lot_sell=lot_sell+lot;
            ask_opt=ask_opt+price*lot;
            bid_opt=bid_opt+(price-spread)*lot;
           }
        }
     }

   if(lot_buy>0 || lot_sell>0)//there are open positions
     {
      ask_opt=ask_opt/(lot_buy+lot_sell);
      bid_opt=bid_opt/(lot_buy+lot_sell);

      bid_opt=NormalizeDouble((ask_opt+bid_opt-spread)/2,_Digits);

      double bid=SymbolInfoDouble(_Symbol,SYMBOL_BID);

      if(lot_buy==lot_sell)
         SLNeutral=bid_opt;

      if(lot_buy>lot_sell)
        {
         double min_sl=NormalizeDouble(bid_opt+Slippage*_Point,_Digits),
                new_sl=NormalizeDouble(bid-buy_sl*_Point,_Digits);

         if(SLBuy==0 && new_sl>=min_sl)
            SLBuy=min_sl;

         if(SLBuy>0 && new_sl>SLBuy)
            SLBuy=new_sl;
        }

      if(lot_buy<lot_sell)
        {
         double min_sl=NormalizeDouble(bid_opt-Slippage*_Point,_Digits),
                new_sl=NormalizeDouble(bid+sell_sl*_Point,_Digits);

         if(SLSell==0 && new_sl<=min_sl)
            SLSell=min_sl;

         if(SLSell>0 && new_sl<SLSell)
            SLSell=new_sl;
        }
     }
//---
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void METS()
  {
//---
   int arrup[],sizeup=ArrayCopy(arrup,up);
   int arrdn[],sizedn=ArrayCopy(arrdn,dn);
   int stoplvl=(int)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL);
   double freezelvl=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_FREEZE_LEVEL)*_Point,mintp=stoplvl*_Point;

   for(int i=sizeup-2; i>=0; i--)
      arrup[i]=arrup[i]+arrup[i+1];

   for(int i=sizedn-2; i>=0; i--)
      arrdn[i]=arrdn[i]+arrdn[i+1];

   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      ulong ticket=PositionGetTicket(i);
      if(PositionSelectByTicket(ticket)==true)
        {
         double lot=PositionGetDouble(POSITION_VOLUME)*pointvalue,
                profitpoint=(PositionGetDouble(POSITION_PROFIT)+PositionGetDouble(POSITION_SWAP)-PositionGetDouble(POSITION_COMMISSION))/lot;

         int indx=(int)MathFloor(profitpoint);

         if(indx>stoplvl)
           {
            bool modify=false;
            int _sl=0,_tp=0;
            double open=PositionGetDouble(POSITION_PRICE_OPEN),
                   price=PositionGetDouble(POSITION_PRICE_CURRENT),
                   sl=PositionGetDouble(POSITION_SL),
                   tp=PositionGetDouble(POSITION_TP),
                   mr=-DBL_MAX;

            if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
              {
               indx=MathMin(indx,sizedn-1);
               for(int i=stoplvl; i<sizeup; i++)
                  for(int j=stoplvl; j<=indx; j++)
                    {
                     double prob=arrup[i]*(arrdn[0]-arrdn[j]);
                     prob=prob/(prob+arrdn[j]*(arrup[0]-arrup[i]));
                     double curmr=MathPow(profitpoint+i,prob)*MathPow(profitpoint-j,1-prob)-profitpoint;
                     if(curmr>profitpoint && curmr>mr)
                       {
                        mr=curmr;
                        _tp=i;
                        _sl=j;
                       }
                    }

               if(mr>0 && profitpoint>_sl+Slippage)
                 {
                  double new_sl=NormalizeDouble(price-_sl*_Point,_Digits);

                  if(sl==0)
                     sl=open;

                  if(price-sl>freezelvl && new_sl>sl)
                    {
                     modify=true;
                     sl=new_sl;
                    }

                  if(UseTakeProfit==true && sl>open)
                    {
                     double new_tp=NormalizeDouble(price+_tp*_Point,_Digits);

                     if(tp==0)
                        tp=NormalizeDouble(price+mintp,_Digits);

                     if(tp-price>freezelvl  && new_tp>tp)
                       {
                        modify=true;
                        tp=new_tp;
                       }
                    }
                 }
              }
            else
              {
               indx=MathMin(indx,sizeup-1);
               for(int i=stoplvl; i<sizedn; i++)
                  for(int j=stoplvl; j<=indx; j++)
                    {
                     double prob=arrdn[i]*(arrup[0]-arrup[j]);
                     prob=prob/(prob+arrup[j]*(arrdn[0]-arrdn[i]));
                     double curmr=MathPow(profitpoint+i,prob)*MathPow(profitpoint-j,1-prob)-profitpoint;
                     if(curmr>profitpoint && curmr>mr)
                       {
                        mr=curmr;
                        _tp=i;
                        _sl=j;
                       }
                    }

               if(mr>0 && profitpoint>_sl+Slippage)
                 {
                  double new_sl=NormalizeDouble(price+_sl*_Point,_Digits);

                  if(sl==0)
                     sl=open;

                  if(sl-price>freezelvl && new_sl<sl)
                    {
                     modify=true;
                     sl=new_sl;
                    }

                  if(UseTakeProfit==true && sl<open)
                    {
                     double new_tp=NormalizeDouble(price-_tp*_Point,_Digits);

                     if(tp==0)
                        tp=NormalizeDouble(price-mintp,_Digits);

                     if(price-tp>freezelvl && new_tp<tp)
                       {
                        modify=true;
                        tp=new_tp;
                       }
                    }
                 }
              }

            if(modify==true && trade.PositionModify(ticket,sl,tp)==false)
               Print("Modification error ",GetLastError());
           }
        }
     }
//---
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SimpleTS()
  {
//---
   double freezelvl=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_FREEZE_LEVEL)*_Point,
          mintp=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL)*_Point;

   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      ulong ticket=PositionGetTicket(i);
      if(PositionSelectByTicket(ticket)==true)
        {
         bool modify=false;
         double open=PositionGetDouble(POSITION_PRICE_OPEN),
                price=PositionGetDouble(POSITION_PRICE_CURRENT),
                sl=PositionGetDouble(POSITION_SL),
                tp=PositionGetDouble(POSITION_TP),
                swap=PositionGetDouble(POSITION_SWAP)-PositionGetDouble(POSITION_COMMISSION),
                lot=PositionGetDouble(POSITION_VOLUME)*pointvalue;

         if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
           {
            double min_sl=NormalizeDouble(open-MathFloor(swap/lot+Slippage)*_Point,_Digits),
                   new_sl=NormalizeDouble(price-buy_sl*_Point,_Digits);

            if(sl==0)
               sl=min_sl;

            sl=MathMax(sl,min_sl);

            if(price-sl>freezelvl && new_sl>sl)
              {
               modify=true;
               sl=new_sl;
              }

            if(UseTakeProfit==true && sl>min_sl)//take profit can be modified
              {
               double new_tp=NormalizeDouble(price+buy_tp*_Point,_Digits);

               if(tp==0)
                  tp=NormalizeDouble(price+mintp,_Digits);

               if(tp-price>freezelvl  && new_tp>tp)
                 {
                  modify=true;
                  tp=new_tp;
                 }
              }
           }
         else
           {
            double min_sl=NormalizeDouble(open+MathCeil(swap/lot-Slippage)*_Point,_Digits),
                   new_sl=NormalizeDouble(price+sell_sl*_Point,_Digits);

            if(sl==0)
               sl=min_sl;

            sl=MathMin(sl,min_sl);

            if(sl-price>freezelvl && new_sl<sl)
              {
               modify=true;
               sl=new_sl;
              }

            if(UseTakeProfit==true && sl<min_sl)
              {
               double new_tp=NormalizeDouble(price-sell_tp*_Point,_Digits);

               if(tp==0)
                  tp=NormalizeDouble(price-mintp,_Digits);

               if(price-tp>freezelvl && new_tp<tp)
                 {
                  modify=true;
                  tp=new_tp;
                 }
              }
           }

         if(modify==true && trade.PositionModify(ticket,sl,tp)==false)
            Print("Modification error ",GetLastError());
        }
     }
//---
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CalcLvl(int &array[],int value)
  {
//---
   int s=ArraySize(array);
   if(s>value)
      array[value]++;
   else
     {
      int a[];
      ArrayResize(a,value+1);
      ArrayInitialize(a,0);
      for(int i=0; i<s; i++)
         a[i]=array[i];
      a[value]=value;
      ArrayCopy(array,a);
     }
//---
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CalcSL(int &array[])
  {
//---
   int a[],s=ArrayCopy(a,array),sl=0;
   ulong max=0;

   for(int i=s-2; i>=0; i--)
      a[i]=a[i]+a[i+1];

   for(int i=0; i<s; i++)
     {
      ulong res=(s-i)*(a[0]-a[i]);
      if(max<res)
        {
         max=res;
         sl=i;
        }
     }
   return((int)MathMax(sl,SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL)));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CalcTP(int &array[])
  {
//---
   int a[],s=ArrayCopy(a,array),tp=0;
   ulong max=0;

   for(int i=s-2; i>=0; i--)
      a[i]=a[i]+a[i+1];

   for(int i=0; i<s; i++)
     {
      ulong res=i*a[i];
      if(max<res)
        {
         max=res;
         tp=i;
        }
     }
   return((int)MathMax(tp,SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL)));
//---
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool NewBarTS()
  {
//---
   static long lastbar;
   long curbar=SeriesInfoInteger(_Symbol,TFTS,SERIES_LASTBAR_DATE);

   if(lastbar<curbar)
     {
      lastbar=curbar;
      return(true);
     }
   return(false);
//---
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool NewBar()
  {
//---
   static long lastbar;
   long curbar=SeriesInfoInteger(_Symbol,PERIOD_CURRENT,SERIES_LASTBAR_DATE);

   if(lastbar<curbar)
     {
      lastbar=curbar;
      return(true);
     }
   return(false);
//---
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Check if there is enough money to open a trade                   |
//+------------------------------------------------------------------+
bool CheckMoneyForTrade(int orderType, double XxX)
  {
   double freeMargin = AccountInfoDouble(ACCOUNT_FREEMARGIN);
   double marginRequired = XxX * SymbolInfoDouble(_Symbol, SYMBOL_MARGIN_INITIAL);
   return (freeMargin > marginRequired);
  }

//+------------------------------------------------------------------+
bool IsMarketClosed()
  {
   datetime currentTime = TimeCurrent();
   MqlDateTime tm;
   TimeToStruct(currentTime, tm);

   int dayOfWeek = tm.day_of_week;
   int hour = tm.hour;

// Verifica si es fin de semana
   if(dayOfWeek <= Sunday || dayOfWeek >= Saturday)
     {
      return true;
     }

// Verifica si está fuera del horario habitual de mercado (ejemplo: 21:00 a 21:59 UTC)
   if(hour >= after || hour < before)  // Ajusta estos valores según el horario del mercado
     {
      return true;
     }

   return false;
  }
//+------------------------------------------------------------------+
double get_lot(double price)
  {
   if(inp_lot_type==LOT_TYPE_FIX)
      return(normalize_lot(inp_lot_fix));
   double one_lot_margin;
   if(!OrderCalcMargin(ORDER_TYPE_BUY,_Symbol,1.0,price,one_lot_margin))
      return(inp_lot_fix);
   return(normalize_lot((AccountInfoDouble(ACCOUNT_BALANCE)*(inp_lot_risk/100))/ one_lot_margin));
  }
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
double normalize_lot(double lt)
  {
   double lot_step = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   lt = MathFloor(lt / lot_step) * lot_step;
   double lot_minimum = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   lt = MathMax(lt, lot_minimum);
   return(lt);
  }
//+------------------------------------------------------------------+
int Vol_strat(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order, double adx)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==1)
     {
      if(heikCol ==1.0)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==0)
     {

      if(heikCol==1.0)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==8)
     {

      if(heikCol==1.0)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0)
           {
            BUYWithOutOrders(Ask);
           }
     }

   if(PositionsTotal() == 0 && ADX_threshold_under  < adx < ADX_threshold_over && inp_stops_type==4)
     {
      if(heikCol ==1.0)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }


   if(PositionsTotal() == 0 && ADX_threshold_under  < adx < ADX_threshold_over && inp_stops_type==5)
     {

      if(heikCol==1.0)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0 && ADX_threshold_under  < adx < ADX_threshold_over && Volumen > VolumeThreshold && inp_stops_type==6)
     {
      if(heikCol ==1.0)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }


   if(PositionsTotal() == 0 && ADX_threshold_under  < adx < ADX_threshold_over && Volumen > VolumeThreshold && inp_stops_type==7)
     {

      if(heikCol==1.0)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   return(Order=0);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int Vol_strat2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order, double adx)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;



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
      return(variable_inutil=0);
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float closes=(float)iClose(_Symbol,PERIOD_D1,0);
   if(ExtMin>closes)
      ExtMin=closes;
   if(ExtMax<closes)
      ExtMax=closes;


//--- predict next price
   PredictPrice();


     {

      if(PositionsTotal() == 0 && heikCol ==1.0 && Order==-1 && inp_stops_type==1 && lastPredicted > predicted)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(PositionsTotal() == 0 && heikCol==0.0 && Order==1 && inp_stops_type==1 && lastPredicted < predicted)
           {
            BUYwithPIPs(Bid, Ask);
           }

      if(PositionsTotal() == 0 && inp_stops_type==0)
        {

         if(heikCol==1.0 && Order==-1 && lastPredicted > predicted)
           {
            SELLwithATR(Bid, entryPrice, digits);
           }
         else
            if(heikCol==0.0 && Order==1 && lastPredicted < predicted)
              {
               BUYwithATR(Ask, entryPrice, digits);
              }
        }

      if(PositionsTotal() == 0 && inp_stops_type==2)
        {

         if(heikCol==1.0 && Order==-1 && lastPredicted > predicted)
           {
            SELLwithPIPs(Bid, Ask);
           }
         else
            if(heikCol==0.0 && Order==1 && lastPredicted < predicted)
              {
               BUYwithPIPs(Bid, Ask);
              }



        }
      if(PositionsTotal() == 0 && inp_stops_type==3)
        {

         if(heikCol ==1.0 && Order==-1 && lastPredicted > predicted)
           {
            SELLwithATR(Bid, entryPrice, digits);
           }
         else
            if(heikCol==0.0 && Order==1 && lastPredicted < predicted)
              {
               BUYwithATR(Ask, entryPrice, digits);
              }
        }
      if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==8)
        {

         if(heikCol==1.0 && lastPredicted > predicted)
           {
            SELLWithOutOrders(Bid);
           }
         else
            if(heikCol==0.0 && lastPredicted < predicted)
              {
               BUYWithOutOrders(Ask);
              }
        }

      if(PositionsTotal() == 0 && ADX_threshold_under  < adx < ADX_threshold_over && inp_stops_type==4)
        {
         if(heikCol ==1.0 && lastPredicted > predicted)
           {
            SELLwithPIPs(Bid, Ask);
           }
         else
            if(heikCol==0.0 && lastPredicted < predicted)
              {
               BUYwithPIPs(Bid, Ask);
              }
        }


      if(PositionsTotal() == 0 && ADX_threshold_under  < adx < ADX_threshold_over && inp_stops_type==5)
        {

         if(heikCol==1.0 && lastPredicted > predicted)
           {
            SELLwithATR(Bid, entryPrice, digits);
           }
         else
            if(heikCol==0.0 && lastPredicted < predicted)
              {
               BUYwithATR(Ask, entryPrice, digits);
              }
        }


      if(PositionsTotal() == 0 && ADX_threshold_under  < adx < ADX_threshold_over && Volumen > VolumeThreshold && inp_stops_type==6)
        {
         if(heikCol ==1.0 && lastPredicted > predicted)
           {
            SELLwithPIPs(Bid, Ask);
           }
         else
            if(heikCol==0.0 && lastPredicted < predicted)
              {
               BUYwithPIPs(Bid, Ask);
              }
        }


      if(PositionsTotal() == 0 && ADX_threshold_under  < adx < ADX_threshold_over && Volumen > VolumeThreshold && inp_stops_type==7)
        {

         if(heikCol==1.0 && lastPredicted > predicted)
           {
            SELLwithATR(Bid, entryPrice, digits);
           }
         else
            if(heikCol==0.0 && lastPredicted < predicted)
              {
               BUYwithATR(Ask, entryPrice, digits);
              }
        }






     }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
   return(Order=0);
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
//Print("Predicted ",predicted, " last predicted: ", lastPredicted);
   if((lastPredicted-predicted)>0)
     {
      Print("Pred says Up");
      Order=1;
     }
   if((lastPredicted-predicted)<0)
     {
      Print("Pred says Down");
      Order=-1;
     }
//m_chart_h_line.Create(0,"Close",0,predicted);  // -delta00);  // set price to 0 to hide the line
//m_chart_h_line.Color(clrDarkGoldenrod);
//--- classify predicted price movement
   return;//(Order);
  }

//+------------------------------------------------------------------+
int MAs(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits,int Order, double Fast, double Slow, double adx)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   Print(atr_val1[0]);
   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0 && Fast > Slow)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && Fast < Slow)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0 && Fast > Slow)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && Fast < Slow)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0 && Fast > Slow)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && Fast < Slow)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0 && Fast > Slow)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && Fast < Slow)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==8)
     {

      if(heikCol==1.0 && Fast > Slow)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0 && Fast < Slow)
           {
            BUYWithOutOrders(Ask);
           }
     }
   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0 && Fast > Slow)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && Fast < Slow)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0 && Fast > Slow)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && Fast < Slow)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0 && Fast > Slow)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && Fast < Slow)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0 && Fast > Slow)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && Fast < Slow)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   return(Order=0);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int MAs2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order, double Fast, double Slow, double adx)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];
   ArraySetAsSeries(atr_val1, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;


   if(PositionsTotal() == 0)
     {

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
         return(variable_inutil=0);
      //--- set next bar time
      ExtNextBar=TimeCurrent();
      ExtNextBar-=ExtNextBar%PeriodSeconds();
      ExtNextBar+=PeriodSeconds();
      //--- check min and max
      float closes=(float)iClose(_Symbol,PERIOD_D1,0);
      if(ExtMin>closes)
         ExtMin=closes;
      if(ExtMax<closes)
         ExtMax=closes;


      //--- predict next price
      PredictPrice();


        {

         if(PositionsTotal() == 0 &&heikCol ==1.0 && Order==-1 && inp_stops_type==0 && Fast > Slow && lastPredicted > predicted)
           {
            SELLwithATR(Bid, entryPrice, digits);
           }
         else
            if(PositionsTotal() == 0 && heikCol==0.0 && Order==1 && inp_stops_type==0 && Fast < Slow && lastPredicted < predicted)
              {
               BUYwithATR(Ask, entryPrice, digits);
              }

         if(PositionsTotal() == 0 && inp_stops_type==1)
           {

            if(heikCol==1.0 && Order==-1 && Fast > Slow && lastPredicted > predicted)
              {
               SELLwithPIPs(Bid, Ask);
              }
            else
               if(heikCol==0.0 && Order==1 && Fast < Slow && lastPredicted < predicted)
                 {
                  BUYwithPIPs(Bid, Ask);
                 }
           }

         if(PositionsTotal() == 0 && inp_stops_type==2 && vol2 > VolumeThreshold)
           {
            Print("llega aquí?");
            if(heikCol==1.0 && Order==-1 && Fast > Slow && lastPredicted > predicted)
              {
               SELLwithPIPs(Bid, Ask);
              }
            else
               if(heikCol==0.0 && Order==1 && Fast < Slow && lastPredicted < predicted)
                 {
                  BUYwithPIPs(Bid, Ask);
                 }



           }
         if(PositionsTotal() == 0 && inp_stops_type==3 && vol2 > VolumeThreshold)
           {

            if(heikCol ==1.0 && Order==-1 && Fast > Slow && lastPredicted > predicted)
              {
               SELLwithATR(Bid, entryPrice, digits);
              }
            else
               if(heikCol==0.0 && Order==1 && Fast < Slow && lastPredicted < predicted)
                 {
                  BUYwithATR(Ask, entryPrice, digits);
                 }
           }

         if(PositionsTotal() == 0  && inp_stops_type==8)
           {

            if(heikCol==1.0 && Fast > Slow && lastPredicted > predicted)
              {
               SELLWithOutOrders(Bid);
              }
            else
               if(heikCol==0.0 && Fast < Slow && lastPredicted < predicted)
                 {
                  BUYWithOutOrders(Ask);
                 }
           }
         if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
           {
            if(heikCol ==1.0 && Fast > Slow && lastPredicted > predicted)
              {
               SELLwithPIPs(Bid, Ask);
              }
            else
               if(heikCol==0.0 && Fast < Slow && lastPredicted < predicted)
                 {
                  BUYwithPIPs(Bid, Ask);
                 }
           }
         if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
           {

            if(heikCol==1.0 && Fast > Slow && lastPredicted > predicted)
              {
               SELLwithATR(Bid, entryPrice, digits);
              }
            else
               if(heikCol==0.0 && Fast < Slow && lastPredicted < predicted)
                 {
                  BUYwithATR(Ask, entryPrice, digits);
                 }
           }

         if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
           {
            if(heikCol ==1.0 && Fast > Slow && lastPredicted > predicted)
              {
               SELLwithPIPs(Bid, Ask);
              }
            else
               if(heikCol==0.0 && Fast < Slow && lastPredicted < predicted)
                 {
                  BUYwithPIPs(Bid, Ask);
                 }
           }
         if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
           {

            if(heikCol==1.0 && Fast > Slow && lastPredicted > predicted)
              {
               SELLwithATR(Bid, entryPrice, digits);
              }
            else
               if(heikCol==0.0 && Fast < Slow && lastPredicted < predicted)
                 {
                  BUYwithATR(Ask, entryPrice, digits);
                 }
           }
        }

     }
   return(Order=0);
  }
//+------------------------------------------------------------------+
void BUYWithOutOrders(double Ask)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

//double atrValue = atr[0];
   Sl = 0.0;//NormalizeDouble(Ask - StopLossPips*_Point,_Digits);
   Tp = 0.0;//NormalizeDouble(Bid + TakeProfitPips*_Point,_Digits);
   double new_lot = get_lot(tick.ask);

   if(CheckMoneyForTrade(ORDER_TYPE_BUY, new_lot))
     {
      // Abrir una posición de compra
      trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, get_lot(tick.ask), Ask, Sl, Tp, "Buy order opened");
      //Orden = "Buy";
     }
   return;
  }
//+------------------------------------------------------------------+
void SELLWithOutOrders(double Bid)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

//double atrValue = atr[0];
   Sl = 0.0;//NormalizeDouble(Bid + StopLossPips*_Point,_Digits);
   Tp = 0.0;//NormalizeDouble(Ask - TakeProfitPips*_Point,_Digits);
   double new_lot = get_lot(tick.bid);

   if(CheckMoneyForTrade(ORDER_TYPE_SELL, new_lot))
     {
      // Abrir una posición de venta
      trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, get_lot(tick.bid), Bid, Sl, Tp, "Sell order opened");
      //Orden = "Sell";
     }
   return;
  }
//+------------------------------------------------------------------+
void BUYwithATR(double Ask, double entryPrice, int digits)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   double atrValue = atr_val1[0];
   Sl = NormalizeDouble(entryPrice - atrValue * atr_multi, digits);
   Tp = NormalizeDouble(entryPrice + atrValue * atr_multi*2, digits);
   double new_lot = get_lot(tick.ask);

   if(CheckMoneyForTrade(ORDER_TYPE_BUY, new_lot))
     {
      // Abrir una posición de compra
      trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, get_lot(tick.ask), Ask, Sl, Tp, "Buy order opened");
      //Orden = "Buy";
     }
  }
//+------------------------------------------------------------------+
void SELLwithATR(double Bid, double entryPrice,  int digits)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   double atrValue = atr_val1[0];
   Sl = NormalizeDouble(entryPrice + atrValue * atr_multi, digits);
   Tp = NormalizeDouble(entryPrice - atrValue * atr_multi*2, digits);
   double new_lot = get_lot(tick.bid);

   if(CheckMoneyForTrade(ORDER_TYPE_SELL, new_lot))
     {
      // Abrir una posición de venta
      trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, get_lot(tick.bid), Bid, Sl, Tp, "Sell order opened");
      //Orden = "Sell";
     }
  }
//+------------------------------------------------------------------+
void BUYwithPIPs(double Bid, double Ask)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
//double atrValue = atr[0];
   Sl = NormalizeDouble(Ask - StopLossPips*_Point,_Digits);
   Tp = NormalizeDouble(Bid + TakeProfitPips*_Point,_Digits);
   double new_lot = get_lot(tick.ask);

   if(CheckMoneyForTrade(ORDER_TYPE_BUY, new_lot))
     {
      // Abrir una posición de compra
      trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, get_lot(tick.ask), Ask, Sl, Tp, "Buy order opened");
      //Orden = "Buy";
     }
  }
//+------------------------------------------------------------------+
void SELLwithPIPs(double Bid, double Ask)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
//double atrValue = atr[0];
   Sl = NormalizeDouble(Bid + StopLossPips*_Point,_Digits);
   Tp = NormalizeDouble(Ask - TakeProfitPips*_Point,_Digits);
   double new_lot = get_lot(tick.bid);

   if(CheckMoneyForTrade(ORDER_TYPE_SELL, new_lot))
     {
      // Abrir una posición de venta
      trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, get_lot(tick.bid), Bid, Sl, Tp, "Sell order opened");
      //Orden = "Sell";
     }
  }

//+------------------------------------------------------------------+
int CCI(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits,int Order, double cci, double adx)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   Print(atr_val1[0]);

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0 && cci < CCI_Sell_Level)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0 && cci < CCI_Sell_Level)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0 && cci < CCI_Sell_Level)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0 && cci < CCI_Sell_Level)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0 && cci < CCI_Sell_Level)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0 && cci < CCI_Sell_Level)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0 && cci < CCI_Sell_Level)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0 && cci < CCI_Sell_Level)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0  && cci < CCI_Sell_Level)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0  && cci > CCI_Buy_Level)
           {
            BUYWithOutOrders(Ask);
           }
     }





   return(Order=0);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CCI2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order, double cci, double adx)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];
   ArraySetAsSeries(atr_val1, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;

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
      return(variable_inutil=0);
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float closes=(float)iClose(_Symbol,PERIOD_D1,0);
   if(ExtMin>closes)
      ExtMin=closes;
   if(ExtMax<closes)
      ExtMax=closes;


//--- predict next price
   PredictPrice();

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0 && cci < CCI_Sell_Level && lastPredicted > predicted)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level && lastPredicted < predicted)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0 && cci < CCI_Sell_Level && lastPredicted > predicted)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level && lastPredicted < predicted)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0 && cci < CCI_Sell_Level && lastPredicted > predicted)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level && lastPredicted < predicted)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0 && cci < CCI_Sell_Level && lastPredicted > predicted)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level && lastPredicted < predicted)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0 && cci < CCI_Sell_Level && lastPredicted > predicted)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level && lastPredicted < predicted)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0 && cci < CCI_Sell_Level && lastPredicted > predicted)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level && lastPredicted < predicted)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0 && cci < CCI_Sell_Level && lastPredicted > predicted)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level && lastPredicted < predicted)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0 && cci < CCI_Sell_Level && lastPredicted > predicted)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && cci > CCI_Buy_Level && lastPredicted < predicted)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0  && cci < CCI_Sell_Level && lastPredicted > predicted)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0  && cci > CCI_Buy_Level && lastPredicted < predicted)
           {
            BUYWithOutOrders(Ask);
           }
     }

   return(Order=0);
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
int Fibo(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits,int Order, double fibo, double adx, double heikenClose)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   Print(atr_val1[0]);

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0 && heikenClose > fibo)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0 && heikenClose > fibo)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0 && heikenClose > fibo)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0 && heikenClose > fibo)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0 && heikenClose > fibo)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0 && heikenClose > fibo)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0 && heikenClose > fibo)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0 && heikenClose > fibo)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0  && heikenClose > fibo)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0  && heikenClose < fibo)
           {
            BUYWithOutOrders(Ask);
           }
     }





   return(Order=0);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int Fibo2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order, double fibo, double adx, double heikenClose)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];
   ArraySetAsSeries(atr_val1, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;


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
      return(variable_inutil=0);
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float closes=(float)iClose(_Symbol,PERIOD_D1,0);
   if(ExtMin>closes)
      ExtMin=closes;
   if(ExtMax<closes)
      ExtMax=closes;


//--- predict next price
   PredictPrice();

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0 && heikenClose > fibo && lastPredicted > predicted)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo && lastPredicted < predicted)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0 && heikenClose > fibo && lastPredicted > predicted)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo && lastPredicted < predicted)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0 && heikenClose > fibo && lastPredicted > predicted)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo && lastPredicted < predicted)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0 && heikenClose > fibo && lastPredicted > predicted)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo && lastPredicted < predicted)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0 && heikenClose > fibo && lastPredicted > predicted)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo && lastPredicted < predicted)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0 && heikenClose > fibo && lastPredicted > predicted)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo && lastPredicted < predicted)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0 && heikenClose > fibo && lastPredicted > predicted)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo && lastPredicted < predicted)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0 && heikenClose > fibo && lastPredicted > predicted)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0 && heikenClose < fibo && lastPredicted < predicted)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0  && heikenClose > fibo && lastPredicted > predicted)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0  && heikenClose < fibo && lastPredicted < predicted)
           {
            BUYWithOutOrders(Ask);
           }
     }

   return(Order=0);
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
int Trend(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits,int Order, double adx)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   Print(atr_val1[0]);

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0)
           {
            BUYWithOutOrders(Ask);
           }
     }





   return(Order=0);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int Trend2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order,  double adx)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];
   ArraySetAsSeries(atr_val1, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;



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
      return(variable_inutil=0);
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float closes=(float)iClose(_Symbol,PERIOD_D1,0);
   if(ExtMin>closes)
      ExtMin=closes;
   if(ExtMax<closes)
      ExtMax=closes;


//--- predict next price
   PredictPrice();

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0  && lastPredicted > predicted)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0  && lastPredicted > predicted)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0  && lastPredicted > predicted)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0  && lastPredicted > predicted)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0  && lastPredicted > predicted)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0  && lastPredicted > predicted)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0  && lastPredicted > predicted)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0  && lastPredicted > predicted)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0   && lastPredicted > predicted)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0   && lastPredicted < predicted)
           {
            BUYWithOutOrders(Ask);
           }
     }

   return(Order=0);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int S_P(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits,int Order, double adx, double heikClose)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   Print(atr_val1[0]);

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0 && heikClose > Resistance_Level)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && heikClose < Support_Level)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0 && heikClose > Resistance_Level)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && heikClose < Support_Level)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0 && heikClose > Resistance_Level)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && heikClose < Support_Level)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0 && heikClose > Resistance_Level)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && heikClose < Support_Level)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0 && heikClose > Resistance_Level)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && heikClose < Support_Level)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0 && heikClose > Resistance_Level)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && heikClose < Support_Level)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0 && heikClose > Resistance_Level)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && heikClose < Support_Level)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0 && heikClose > Resistance_Level)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && heikClose < Support_Level)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0 && heikClose > Resistance_Level)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0  && heikClose < Support_Level)
           {
            BUYWithOutOrders(Ask);
           }
     }





   return(Order=0);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int S_P2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order,  double adx, double heikClose)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];
   ArraySetAsSeries(atr_val1, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;


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
      return(variable_inutil=0);
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float closes=(float)iClose(_Symbol,PERIOD_D1,0);
   if(ExtMin>closes)
      ExtMin=closes;
   if(ExtMax<closes)
      ExtMax=closes;


//--- predict next price
   PredictPrice();

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0  && lastPredicted > predicted  && heikClose > Resistance_Level)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && heikClose < Support_Level)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0  && lastPredicted > predicted && heikClose > Resistance_Level)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && heikClose < Support_Level)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0  && lastPredicted > predicted && heikClose > Resistance_Level)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && heikClose < Support_Level)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0  && lastPredicted > predicted && heikClose > Resistance_Level)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && heikClose < Support_Level)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && heikClose > Resistance_Level)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && heikClose < Support_Level)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0  && lastPredicted > predicted && heikClose > Resistance_Level)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && heikClose < Support_Level)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && heikClose > Resistance_Level)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && heikClose < Support_Level)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0  && lastPredicted > predicted && heikClose > Resistance_Level)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && heikClose < Support_Level)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0   && lastPredicted > predicted && heikClose > Resistance_Level)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0   && lastPredicted < predicted && heikClose < Support_Level)
           {
            BUYWithOutOrders(Ask);
           }
     }

   return(Order=0);
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int RSI(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits,int Order, double adx, double rsi)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   Print(atr_val1[0]);

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0 && rsi > Oversold)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && rsi < Overbought)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0 && rsi > Oversold)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && rsi < Overbought)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0 && rsi > Oversold)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && rsi < Overbought)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0 && rsi > Oversold)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && rsi < Overbought)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0 && rsi > Oversold)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && rsi < Overbought)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0 && rsi > Oversold)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && rsi < Overbought)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0 && rsi > Oversold)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && rsi < Overbought)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0 && rsi > Oversold)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && rsi < Overbought)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0 && rsi > Oversold)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0  && rsi < Overbought)
           {
            BUYWithOutOrders(Ask);
           }
     }





   return(Order=0);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int RSI2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order,  double adx, double rsi)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];
   ArraySetAsSeries(atr_val1, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;

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
      return(variable_inutil=0);
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float closes=(float)iClose(_Symbol,PERIOD_D1,0);
   if(ExtMin>closes)
      ExtMin=closes;
   if(ExtMax<closes)
      ExtMax=closes;


//--- predict next price
   PredictPrice();

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0  && lastPredicted > predicted  && rsi > Oversold)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && rsi < Overbought)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0  && lastPredicted > predicted && rsi > Oversold)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && rsi < Overbought)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0  && lastPredicted > predicted && rsi > Oversold)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && rsi < Overbought)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0  && lastPredicted > predicted && rsi > Oversold)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && rsi < Overbought)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && rsi > Oversold)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && rsi < Overbought)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0  && lastPredicted > predicted && rsi > Oversold)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && rsi < Overbought)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && rsi > Oversold)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && rsi < Overbought)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0  && lastPredicted > predicted && rsi > Oversold)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && rsi < Overbought)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0   && lastPredicted > predicted && rsi > Oversold)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0   && lastPredicted < predicted && rsi < Overbought)
           {
            BUYWithOutOrders(Ask);
           }
     }

   return(Order=0);
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int Bol(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits,int Order, double adx, double heikClose, double middleBand)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   Print(atr_val1[0]);

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0 && middleBand > heikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && middleBand < heikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0 && middleBand > heikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && middleBand < heikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0 && middleBand > heikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && middleBand < heikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0 && middleBand > heikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && middleBand < heikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0 && middleBand > heikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && middleBand < heikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0 && middleBand > heikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && middleBand < heikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0 && middleBand > heikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && middleBand < heikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0 && middleBand > heikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && middleBand < heikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0 && middleBand > heikClose)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0  && middleBand < heikClose)
           {
            BUYWithOutOrders(Ask);
           }
     }





   return(Order=0);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int Bol2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order,  double adx, double heikClose, double middleBand)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];
   ArraySetAsSeries(atr_val1, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;

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
      return(variable_inutil=0);
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float closes=(float)iClose(_Symbol,PERIOD_D1,0);
   if(ExtMin>closes)
      ExtMin=closes;
   if(ExtMax<closes)
      ExtMax=closes;


//--- predict next price
   PredictPrice();

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0  && lastPredicted > predicted  && middleBand > heikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && middleBand < heikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0  && lastPredicted > predicted && middleBand > heikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && middleBand < heikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0  && lastPredicted > predicted && middleBand > heikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && middleBand < heikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0  && lastPredicted > predicted && middleBand > heikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && middleBand < heikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && middleBand > heikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && middleBand < heikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0  && lastPredicted > predicted && middleBand > heikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && middleBand < heikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && middleBand > heikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && middleBand < heikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0  && lastPredicted > predicted && middleBand > heikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && middleBand < heikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0   && lastPredicted > predicted && middleBand > heikClose)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0   && lastPredicted < predicted && middleBand < heikClose)
           {
            BUYWithOutOrders(Ask);
           }
     }

   return(Order=0);
  }
//+------------------------------------------------------------------+
int MACD(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits,int Order, double adx, double MacdMain, double MacdSignal)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   Print(atr_val1[0]);

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0 && MacdSignal > MacdMain)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && MacdSignal < MacdMain)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0 && MacdSignal > MacdMain)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && MacdSignal < MacdMain)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0 && MacdSignal > MacdMain)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && MacdSignal < MacdMain)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0 && MacdSignal > MacdMain)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && MacdSignal < MacdMain)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0 && MacdSignal > MacdMain)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && MacdSignal < MacdMain)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0 && MacdSignal > MacdMain)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && MacdSignal < MacdMain)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0 && MacdSignal > MacdMain)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && MacdSignal < MacdMain)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0 && MacdSignal > MacdMain)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && MacdSignal < MacdMain)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0 && MacdSignal > MacdMain)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0  && MacdSignal < MacdMain)
           {
            BUYWithOutOrders(Ask);
           }
     }





   return(Order=0);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int MACD2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order,  double adx, double MacdMain, double MacdSignal)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];
   ArraySetAsSeries(atr_val1, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;


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
      return(variable_inutil=0);
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float closes=(float)iClose(_Symbol,PERIOD_D1,0);
   if(ExtMin>closes)
      ExtMin=closes;
   if(ExtMax<closes)
      ExtMax=closes;


//--- predict next price
   PredictPrice();

   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0  && lastPredicted > predicted  && MacdSignal > MacdMain)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && MacdSignal < MacdMain)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0  && lastPredicted > predicted && MacdSignal > MacdMain)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && MacdSignal < MacdMain)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0  && lastPredicted > predicted && MacdSignal > MacdMain)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && MacdSignal < MacdMain)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0  && lastPredicted > predicted && MacdSignal > MacdMain)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && MacdSignal < MacdMain)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && MacdSignal > MacdMain)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && MacdSignal < MacdMain)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0  && lastPredicted > predicted && MacdSignal > MacdMain)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && MacdSignal < MacdMain)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && MacdSignal > MacdMain)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && MacdSignal < MacdMain)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0  && lastPredicted > predicted && MacdSignal > MacdMain)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && MacdSignal < MacdMain)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0   && lastPredicted > predicted && MacdSignal > MacdMain)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0   && lastPredicted < predicted && MacdSignal < MacdMain)
           {
            BUYWithOutOrders(Ask);
           }
     }

   return(Order=0);
  }
//+------------------------------------------------------------------+
int Stoch(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits,int Order, double adx, double KValue, double DValue)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   Print(atr_val1[0]);


   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0 && DValue > KValue && KValue > 80)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && DValue < KValue && KValue < 20)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0 && DValue > KValue && KValue > 80)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && DValue < KValue && KValue < 20)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0 && DValue > KValue && KValue > 80)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && DValue < KValue && KValue < 20)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0 && DValue > KValue && KValue > 80)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && DValue < KValue && KValue < 20)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0 && DValue > KValue && KValue > 80)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && DValue < KValue && KValue < 20)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0 && DValue > KValue && KValue > 80)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && DValue < KValue && KValue < 20)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0 && DValue > KValue && KValue > 80)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && DValue < KValue && KValue < 20)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0 && DValue > KValue && KValue > 80)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && DValue < KValue && KValue < 20)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0 && DValue > KValue && KValue > 80)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0  && DValue < KValue && KValue < 20)
           {
            BUYWithOutOrders(Ask);
           }
     }





   return(Order=0);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int Stoch2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order,  double adx, double KValue, double DValue)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];
   ArraySetAsSeries(atr_val1, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;



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
      return(variable_inutil=0);
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float closes=(float)iClose(_Symbol,PERIOD_D1,0);
   if(ExtMin>closes)
      ExtMin=closes;
   if(ExtMax<closes)
      ExtMax=closes;


//--- predict next price
   PredictPrice();



   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      Print("esto va bien?");
      if(heikCol ==1.0  && lastPredicted > predicted  && DValue > KValue && KValue > 80)
        {
         Print("esto va bien2?");
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && DValue < KValue && KValue < 20)
           {
            Print("esto va bien3?");
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0  && lastPredicted > predicted && DValue > KValue && KValue > 80)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && DValue < KValue && KValue < 20)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0  && lastPredicted > predicted && DValue > KValue && KValue > 80)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && DValue < KValue && KValue < 20)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0  && lastPredicted > predicted && DValue > KValue && KValue > 80)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && DValue < KValue && KValue < 20)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && DValue > KValue && KValue > 80)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && DValue < KValue && KValue < 20)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0  && lastPredicted > predicted && DValue > KValue && KValue > 80)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && DValue < KValue && KValue < 20)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && DValue > KValue && KValue > 80)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && DValue < KValue && KValue < 20)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0  && lastPredicted > predicted && DValue > KValue && KValue > 80)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && DValue < KValue && KValue < 20)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(inp_stops_type==8 && PositionsTotal() == 0)
     {

      if(heikCol==1.0   && lastPredicted > predicted && DValue > KValue && KValue > 80)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0   && lastPredicted < predicted && DValue < KValue && KValue < 20)
           {
            BUYWithOutOrders(Ask);
           }
     }


   return(Order=0);
  }



//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int PSAR(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits,int Order, double adx, double HeikClose, double psar)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   Print(atr_val1[0]);

   Print(Volumen);


   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0 && psar > HeikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && psar < HeikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0 && psar > HeikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && psar < HeikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0 && psar > HeikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && psar < HeikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0 && psar > HeikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && psar < HeikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0 && psar > HeikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && psar < HeikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0 && psar > HeikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && psar < HeikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0 && psar > HeikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && psar < HeikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0 && psar > HeikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && psar < HeikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0 && psar > HeikClose)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0  && psar < HeikClose)
           {
            BUYWithOutOrders(Ask);
           }
     }





   return(Order=0);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int PSAR2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order,  double adx, double HeikClose, double psar)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];
   ArraySetAsSeries(atr_val1, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;

   Print(Volumen);


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
      return(variable_inutil=0);
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float closes=(float)iClose(_Symbol,PERIOD_D1,0);
   if(ExtMin>closes)
      ExtMin=closes;
   if(ExtMax<closes)
      ExtMax=closes;


//--- predict next price
   PredictPrice();





   if(PositionsTotal() == 0  && inp_stops_type==1)
     {

      if(heikCol ==1.0  && lastPredicted > predicted  && psar > HeikClose)
        {

         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && psar < HeikClose)
           {

            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0  && lastPredicted > predicted && psar > HeikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && psar < HeikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0  && lastPredicted > predicted && psar > HeikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && psar < HeikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0  && lastPredicted > predicted && psar > HeikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && psar < HeikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && psar > HeikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && psar < HeikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0  && lastPredicted > predicted && psar > HeikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && psar < HeikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && psar > HeikClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && psar < HeikClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }

   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0  && lastPredicted > predicted && psar > HeikClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && psar < HeikClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(inp_stops_type==8 && PositionsTotal() == 0)
     {

      if(heikCol==1.0   && lastPredicted > predicted && psar > HeikClose)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0   && lastPredicted < predicted && psar < HeikClose)
           {
            BUYWithOutOrders(Ask);
           }
     }

   return(Order=0);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int REN(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits,int Order, double adx, bool maRising, double maFalling)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   Print(atr_val1[0]);

   Print(Volumen);

   double renkoClose = lastRenkoClose + (maRising ? 1 : (maFalling ? -1 : 0));


   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0 && maFalling == true)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && maRising == true)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0 && maFalling == true)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && maRising == true)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0 && maFalling == true)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && maRising == true)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0 && maFalling == true)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && maRising == true)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0 && maFalling == true)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && maRising == true)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0 && maFalling == true)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && maRising == true)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0 && maFalling == true)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && maRising == true)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0 && maFalling == true)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && maRising == true)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0 && maFalling == true)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0  && maRising == true)
           {
            BUYWithOutOrders(Ask);
           }
     }

   return(lastRenkoClose = renkoClose);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int REN2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order,  double adx, bool maRising, bool maFalling)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];
   ArraySetAsSeries(atr_val1, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;

   Print(Volumen);

   double renkoClose = lastRenkoClose + (maRising ? 1 : (maFalling ? -1 : 0));

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
      return(variable_inutil=0);
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float closes=(float)iClose(_Symbol,PERIOD_D1,0);
   if(ExtMin>closes)
      ExtMin=closes;
   if(ExtMax<closes)
      ExtMax=closes;

//--- predict next price
   PredictPrice();


   if(PositionsTotal() == 0  && inp_stops_type==1)
     {

      if(heikCol ==1.0  && lastPredicted > predicted  && maFalling == true)
        {

         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true)
           {

            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0  && lastPredicted > predicted && maFalling == true)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0  && lastPredicted > predicted && maFalling == true)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0  && lastPredicted > predicted && maFalling == true)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && maFalling == true)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0  && lastPredicted > predicted && maFalling == true)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && maFalling == true)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }

   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0  && lastPredicted > predicted && maFalling == true)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(inp_stops_type==8 && PositionsTotal() == 0)
     {

      if(heikCol==1.0   && lastPredicted > predicted && maFalling == true)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0   && lastPredicted < predicted && maRising == true)
           {
            BUYWithOutOrders(Ask);
           }
     }

   return(lastRenkoClose = renkoClose);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int RENB(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits,int Order, double adx, bool maRising, double maFalling)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   Print(atr_val1[0]);

   Print(Volumen);

   double renkoClose = lastRenkoClose + (maRising ? 1 : (maFalling ? -1 : 0));


   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(heikCol ==1.0 && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0 && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0 && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0 && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0 && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0 && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0 && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0 && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0 && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(heikCol==1.0 && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0  && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYWithOutOrders(Ask);
           }
     }

   return(lastRenkoClose = renkoClose);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int RENB2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order,  double adx, bool maRising, bool maFalling)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];
   ArraySetAsSeries(atr_val1, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;

   Print(Volumen);

   double renkoClose = lastRenkoClose + (maRising ? 1 : (maFalling ? -1 : 0));

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
      return(variable_inutil=0);
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float closes=(float)iClose(_Symbol,PERIOD_D1,0);
   if(ExtMin>closes)
      ExtMin=closes;
   if(ExtMax<closes)
      ExtMax=closes;

//--- predict next price
   PredictPrice();


   if(PositionsTotal() == 0  && inp_stops_type==1)
     {

      if(heikCol ==1.0  && lastPredicted > predicted  && maFalling == true && renkoClose < lastRenkoClose)
        {

         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true && renkoClose > lastRenkoClose)
           {

            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(heikCol==1.0  && lastPredicted > predicted && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(heikCol==1.0  && lastPredicted > predicted && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(heikCol ==1.0  && lastPredicted > predicted && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(heikCol==1.0  && lastPredicted > predicted && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(heikCol ==1.0  && lastPredicted > predicted && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }

   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(heikCol==1.0  && lastPredicted > predicted && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(heikCol==0.0  && lastPredicted < predicted && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(inp_stops_type==8 && PositionsTotal() == 0)
     {

      if(heikCol==1.0   && lastPredicted > predicted && maFalling == true && renkoClose < lastRenkoClose)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(heikCol==0.0   && lastPredicted < predicted && maRising == true && renkoClose > lastRenkoClose)
           {
            BUYWithOutOrders(Ask);
           }
     }

   return(lastRenkoClose = renkoClose);
  }



//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void WIS(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits,int Order, double adx, double ma00, double ma01,double ma10, double ma11)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);
   double atr_val1[];

   ArraySetAsSeries(atr_val1, true);

   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);
   Print(atr_val1[0]);





   if(PositionsTotal() == 0  && inp_stops_type==1)
     {
      if(ma00 < ma10 && ma01 > ma11)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(ma00 > ma10 && ma01 < ma11)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(ma00 < ma10 && ma01 > ma11)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(ma00 > ma10 && ma01 < ma11)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(ma00 < ma10 && ma01 > ma11)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(ma00 > ma10 && ma01 < ma11)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(ma00 < ma10 && ma01 > ma11)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(ma00 > ma10 && ma01 < ma11)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(ma00 < ma10 && ma01 > ma11)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(ma00 > ma10 && ma01 < ma11)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(ma00 < ma10 && ma01 > ma11)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(ma00 > ma10 && ma01 < ma11)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(ma00 < ma10 && ma01 > ma11)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(ma00 > ma10 && ma01 < ma11)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(ma00 < ma10 && ma01 > ma11)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(ma00 > ma10 && ma01 < ma11)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && inp_stops_type==8)
     {

      if(ma00 < ma10 && ma01 > ma11)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(ma00 > ma10 && ma01 < ma11)
           {
            BUYWithOutOrders(Ask);
           }
     }

  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void WIS2(double entryPrice, double heikCol, int Volumen, double Ask, double Bid, double atr_val, int digits, int Order,  double adx, double ma00, double ma01, double ma10, double ma11)
  {
   MqlTick tick;
   SymbolInfoTick(_Symbol,tick);

   double atr_val1[];
   ArraySetAsSeries(atr_val1, true);
   CopyBuffer(ATRHandle, 0, 0, 2, atr_val1);

   double vol2 = Volumen;





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
   float closes=(float)iClose(_Symbol,PERIOD_D1,0);
   if(ExtMin>closes)
      ExtMin=closes;
   if(ExtMax<closes)
      ExtMax=closes;

//--- predict next price
   PredictPrice();


   if(PositionsTotal() == 0  && inp_stops_type==1)
     {

      if(lastPredicted > predicted  && ma00 < ma10 && ma01 > ma11)
        {

         SELLwithPIPs(Bid, Ask);
        }
      else
         if(lastPredicted < predicted && ma00 > ma10 && ma01 < ma11)
           {

            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==0)
     {

      if(lastPredicted > predicted && ma00 < ma10 && ma01 > ma11)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(lastPredicted < predicted && ma00 > ma10 && ma01 < ma11)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==2)
     {

      if(lastPredicted > predicted && ma00 < ma10 && ma01 > ma11)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(lastPredicted < predicted && ma00 > ma10 && ma01 < ma11)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }



   if(PositionsTotal() == 0 && Volumen > VolumeThreshold && inp_stops_type==3)
     {

      if(lastPredicted > predicted && ma00 < ma10 && ma01 > ma11)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(lastPredicted < predicted && ma00 > ma10 && ma01 < ma11)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(PositionsTotal() == 0  && inp_stops_type==4 && ADX_threshold_under < adx < ADX_threshold_over)
     {
      if(lastPredicted > predicted && ma00 < ma10 && ma01 > ma11)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(lastPredicted < predicted && ma00 > ma10 && ma01 < ma11)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }
   if(PositionsTotal() == 0 &&  inp_stops_type==5 && ADX_threshold_under < adx < ADX_threshold_over)
     {

      if(lastPredicted > predicted && ma00 < ma10 && ma01 > ma11)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(lastPredicted < predicted && ma00 > ma10 && ma01 < ma11)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }


   if(PositionsTotal() == 0  && inp_stops_type==6 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {
      if(lastPredicted > predicted && ma00 < ma10 && ma01 > ma11)
        {
         SELLwithPIPs(Bid, Ask);
        }
      else
         if(lastPredicted < predicted && ma00 > ma10 && ma01 < ma11)
           {
            BUYwithPIPs(Bid, Ask);
           }
     }

   if(PositionsTotal() == 0 &&  inp_stops_type==7 && ADX_threshold_under < adx < ADX_threshold_over && Volumen > VolumeThreshold)
     {

      if(lastPredicted > predicted && ma00 < ma10 && ma01 > ma11)
        {
         SELLwithATR(Bid, entryPrice, digits);
        }
      else
         if(lastPredicted < predicted && ma00 > ma10 && ma01 < ma11)
           {
            BUYwithATR(Ask, entryPrice, digits);
           }
     }

   if(inp_stops_type==8 && PositionsTotal() == 0)
     {

      if(lastPredicted > predicted && ma00 < ma10 && ma01 > ma11)
        {
         SELLWithOutOrders(Bid);
        }
      else
         if(lastPredicted < predicted && ma00 > ma10 && ma01 < ma11)
           {
            BUYWithOutOrders(Ask);
           }
     }


  }
//+------------------------------------------------------------------+
