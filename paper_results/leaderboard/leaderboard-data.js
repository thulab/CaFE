window.CAFE_LEADERBOARD_DATA = {
  "schemaVersion": "cafe.public_leaderboard.v1",
  "lowerIsBetter": true,
  "suiteAggregation": "equal weight over available benchmark suites",
  "capabilityAggregation": "equal weight over levels within suite, then equal weight over available suites",
  "models": [
    "Chronos-2",
    "timesfm2.5",
    "tirex2",
    "moirai2",
    "Timer-3.5",
    "toto2.0"
  ],
  "suites": [
    {
      "id": "all",
      "label": "All benchmarks"
    },
    {
      "id": "GIFT-Short",
      "label": "GIFT-Short"
    },
    {
      "id": "GIFT-Medium",
      "label": "GIFT-Medium"
    },
    {
      "id": "GIFT-Long",
      "label": "GIFT-Long"
    },
    {
      "id": "FEV-Mini20",
      "label": "FEV-Mini20"
    }
  ],
  "capabilities": [
    {
      "id": "trend",
      "label": "Trend"
    },
    {
      "id": "multi_seasonal",
      "label": "Multi-seasonal"
    },
    {
      "id": "time_varying_seasonality",
      "label": "Time-varying seasonality"
    },
    {
      "id": "regime_switching",
      "label": "Regime switching"
    },
    {
      "id": "predictable_intermittency",
      "label": "Predictable intermittency"
    },
    {
      "id": "common_factor",
      "label": "Common factor"
    },
    {
      "id": "cross_series_dependence",
      "label": "Cross-series dependence"
    },
    {
      "id": "covariate_impulse_response",
      "label": "Covariate response"
    }
  ],
  "metrics": {
    "reference_mase": {
      "label": "Reference MASE",
      "shortLabel": "Reference MASE",
      "description": "Forecasting accuracy on the authentic benchmark futures."
    },
    "probe_mase": {
      "label": "Diagnostic-probe MASE",
      "shortLabel": "Probe MASE",
      "description": "Forecasting accuracy on the treated futures."
    },
    "paired_nrmse": {
      "label": "Paired forecast-change NRMSE",
      "shortLabel": "Paired NRMSE",
      "description": "Normalized error in the forecast change induced by each treatment."
    }
  },
  "overall": {
    "GIFT-Short": [
      {
        "model": "Chronos-2",
        "values": {
          "reference_mase": 1.695633754,
          "probe_mase": 2.2263538282,
          "paired_nrmse": 0.533096788
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "timesfm2.5",
        "values": {
          "reference_mase": 1.601633415,
          "probe_mase": 2.1692205555,
          "paired_nrmse": 0.5799871313
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "tirex2",
        "values": {
          "reference_mase": 1.5878163,
          "probe_mase": 2.1510475574,
          "paired_nrmse": 0.6046927472
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "moirai2",
        "values": {
          "reference_mase": 1.821645488,
          "probe_mase": 2.9949659436,
          "paired_nrmse": 0.643338954
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "Timer-3.5",
        "values": {
          "reference_mase": 1.647942927,
          "probe_mase": 2.3137977642,
          "paired_nrmse": 0.6010097051
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "toto2.0",
        "values": {
          "reference_mase": 1.572560429,
          "probe_mase": 2.0684667468,
          "paired_nrmse": 0.6166760812
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      }
    ],
    "GIFT-Medium": [
      {
        "model": "Chronos-2",
        "values": {
          "reference_mase": 2.266452253,
          "probe_mase": 4.2914606129,
          "paired_nrmse": 0.6471411338
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "timesfm2.5",
        "values": {
          "reference_mase": 2.307843953,
          "probe_mase": 3.9620333825,
          "paired_nrmse": 0.7002753986
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "moirai2",
        "values": {
          "reference_mase": 3.082946512,
          "probe_mase": 6.6715390684,
          "paired_nrmse": 0.8821604501
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "Timer-3.5",
        "values": {
          "reference_mase": 2.299305469,
          "probe_mase": 4.6367589864,
          "paired_nrmse": 0.8212944368
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "toto2.0",
        "values": {
          "reference_mase": 5.011916189,
          "probe_mase": 10.4382960796,
          "paired_nrmse": 0.858162133
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      }
    ],
    "GIFT-Long": [
      {
        "model": "Chronos-2",
        "values": {
          "reference_mase": 2.813510424,
          "probe_mase": 4.7988797798,
          "paired_nrmse": 0.5784363042
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "timesfm2.5",
        "values": {
          "reference_mase": 2.920054734,
          "probe_mase": 4.8392823237,
          "paired_nrmse": 0.6738117791
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "moirai2",
        "values": {
          "reference_mase": 4.138173509,
          "probe_mase": 7.3903063915,
          "paired_nrmse": 0.9554177392
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "Timer-3.5",
        "values": {
          "reference_mase": 2.818207779,
          "probe_mase": 5.7680149124,
          "paired_nrmse": 0.7751514368
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "toto2.0",
        "values": {
          "reference_mase": 12.3428511,
          "probe_mase": 23.2674592381,
          "paired_nrmse": 0.8336085494
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      }
    ],
    "FEV-Mini20": [
      {
        "model": "Chronos-2",
        "values": {
          "reference_mase": 1.827602821,
          "probe_mase": 2.1862173952,
          "paired_nrmse": 0.6396533393
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "timesfm2.5",
        "values": {
          "reference_mase": 1.43415681,
          "probe_mase": 1.8568660322,
          "paired_nrmse": 0.5671737159
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "tirex2",
        "values": {
          "reference_mase": 1.746164893,
          "probe_mase": 2.121391467,
          "paired_nrmse": 0.5819525612
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "moirai2",
        "values": {
          "reference_mase": 1.554739182,
          "probe_mase": 2.1305110262,
          "paired_nrmse": 0.6439230115
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "Timer-3.5",
        "values": {
          "reference_mase": 1.705807044,
          "probe_mase": 2.2555887604,
          "paired_nrmse": 0.6235238135
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      },
      {
        "model": "toto2.0",
        "values": {
          "reference_mase": 1.553063204,
          "probe_mase": 1.7768018221,
          "paired_nrmse": 0.519343402
        },
        "coverage": {
          "reference_mase": 1,
          "probe_mase": 1,
          "paired_nrmse": 1
        },
        "suiteCount": 1
      }
    ],
    "all": [
      {
        "model": "Chronos-2",
        "values": {
          "reference_mase": 2.150799813,
          "probe_mase": 3.375727904,
          "paired_nrmse": 0.5995818913
        },
        "coverage": {
          "reference_mase": 4,
          "probe_mase": 4,
          "paired_nrmse": 4
        },
        "suiteCount": 4
      },
      {
        "model": "timesfm2.5",
        "values": {
          "reference_mase": 2.065922228,
          "probe_mase": 3.2068505735,
          "paired_nrmse": 0.6303120062
        },
        "coverage": {
          "reference_mase": 4,
          "probe_mase": 4,
          "paired_nrmse": 4
        },
        "suiteCount": 4
      },
      {
        "model": "tirex2",
        "values": {
          "reference_mase": 1.6669905965,
          "probe_mase": 2.1362195122,
          "paired_nrmse": 0.5933226542
        },
        "coverage": {
          "reference_mase": 2,
          "probe_mase": 2,
          "paired_nrmse": 2
        },
        "suiteCount": 4
      },
      {
        "model": "moirai2",
        "values": {
          "reference_mase": 2.6493761727,
          "probe_mase": 4.7968306074,
          "paired_nrmse": 0.7812100387
        },
        "coverage": {
          "reference_mase": 4,
          "probe_mase": 4,
          "paired_nrmse": 4
        },
        "suiteCount": 4
      },
      {
        "model": "Timer-3.5",
        "values": {
          "reference_mase": 2.1178158048,
          "probe_mase": 3.7435401059,
          "paired_nrmse": 0.705244848
        },
        "coverage": {
          "reference_mase": 4,
          "probe_mase": 4,
          "paired_nrmse": 4
        },
        "suiteCount": 4
      },
      {
        "model": "toto2.0",
        "values": {
          "reference_mase": 5.1200977305,
          "probe_mase": 9.3877559716,
          "paired_nrmse": 0.7069475414
        },
        "coverage": {
          "reference_mase": 4,
          "probe_mase": 4,
          "paired_nrmse": 4
        },
        "suiteCount": 4
      }
    ]
  },
  "capability": {
    "GIFT-Short": {
      "probe_mase": [
        {
          "model": "Chronos-2",
          "scores": {
            "trend": 2.1321202378,
            "multi_seasonal": 2.369753248,
            "time_varying_seasonality": 1.5256988594,
            "regime_switching": 1.702840248,
            "predictable_intermittency": 2.9289153442,
            "common_factor": 0.6175713506,
            "cross_series_dependence": 2.895496619,
            "covariate_impulse_response": 3.6384347186
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 2.2263538282,
          "suiteCount": 1
        },
        {
          "model": "timesfm2.5",
          "scores": {
            "trend": 1.8980614192,
            "multi_seasonal": 2.4992550912,
            "time_varying_seasonality": 1.462328602,
            "regime_switching": 1.6267963472,
            "predictable_intermittency": 2.8326417602,
            "common_factor": 0.648073183,
            "cross_series_dependence": 2.896064845,
            "covariate_impulse_response": 3.4905431964
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 2.1692205555,
          "suiteCount": 1
        },
        {
          "model": "tirex2",
          "scores": {
            "trend": 1.8430118642,
            "multi_seasonal": 2.531765501,
            "time_varying_seasonality": 1.4836768958,
            "regime_switching": 1.5974502548,
            "predictable_intermittency": 2.745443272,
            "common_factor": 0.6576303055,
            "cross_series_dependence": 2.8648627414,
            "covariate_impulse_response": 3.4845396246
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 2.1510475574,
          "suiteCount": 1
        },
        {
          "model": "moirai2",
          "scores": {
            "trend": 2.1518772868,
            "multi_seasonal": 4.1713330496,
            "time_varying_seasonality": 1.8531841032,
            "regime_switching": 1.9098813756,
            "predictable_intermittency": 4.2113167956,
            "common_factor": 0.6949992116,
            "cross_series_dependence": 3.0599299562,
            "covariate_impulse_response": 5.9072057698
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 2.9949659436,
          "suiteCount": 1
        },
        {
          "model": "Timer-3.5",
          "scores": {
            "trend": 1.966418156,
            "multi_seasonal": 2.7582860688,
            "time_varying_seasonality": 1.7383156672,
            "regime_switching": 1.7168932856,
            "predictable_intermittency": 2.8716237326,
            "common_factor": 0.6437498105,
            "cross_series_dependence": 2.9818021622,
            "covariate_impulse_response": 3.8332932304
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 2.3137977642,
          "suiteCount": 1
        },
        {
          "model": "toto2.0",
          "scores": {
            "trend": 1.8764020094,
            "multi_seasonal": 1.9341391972,
            "time_varying_seasonality": 1.5719739018,
            "regime_switching": 1.5736363846,
            "predictable_intermittency": 2.6986557006,
            "common_factor": 0.624883582,
            "cross_series_dependence": 2.7460188158,
            "covariate_impulse_response": 3.522024383
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 2.0684667468,
          "suiteCount": 1
        }
      ],
      "paired_nrmse": [
        {
          "model": "Chronos-2",
          "scores": {
            "trend": 0.0820825094,
            "multi_seasonal": 0.7794513562,
            "time_varying_seasonality": 0.5294781747,
            "regime_switching": 0.0963850868,
            "predictable_intermittency": 0.9051413851,
            "common_factor": 0.2568654659,
            "cross_series_dependence": 0.9892779551,
            "covariate_impulse_response": 0.6260923707
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.533096788,
          "suiteCount": 1
        },
        {
          "model": "timesfm2.5",
          "scores": {
            "trend": 0.2155851914,
            "multi_seasonal": 0.8016590865,
            "time_varying_seasonality": 0.6794172013,
            "regime_switching": 0.0909547536,
            "predictable_intermittency": 0.8373530645,
            "common_factor": 0.333317783,
            "cross_series_dependence": 1.1866928238,
            "covariate_impulse_response": 0.4949171461
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.5799871313,
          "suiteCount": 1
        },
        {
          "model": "tirex2",
          "scores": {
            "trend": 0.2562364899,
            "multi_seasonal": 0.8441032902,
            "time_varying_seasonality": 0.5802932687,
            "regime_switching": 0.0708317311,
            "predictable_intermittency": 0.9724890378,
            "common_factor": 0.2587081813,
            "cross_series_dependence": 1.0045754189,
            "covariate_impulse_response": 0.8503045596
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.6046927472,
          "suiteCount": 1
        },
        {
          "model": "moirai2",
          "scores": {
            "trend": 0.2149589024,
            "multi_seasonal": 1.0101560638,
            "time_varying_seasonality": 0.7404839685,
            "regime_switching": 0.134159986,
            "predictable_intermittency": 0.9653250416,
            "common_factor": 0.3532650474,
            "cross_series_dependence": 1.0439569855,
            "covariate_impulse_response": 0.6844056371
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.643338954,
          "suiteCount": 1
        },
        {
          "model": "Timer-3.5",
          "scores": {
            "trend": 0.2430601202,
            "multi_seasonal": 0.9456936793,
            "time_varying_seasonality": 0.6743320698,
            "regime_switching": 0.1329371331,
            "predictable_intermittency": 0.9144081966,
            "common_factor": 0.300933994,
            "cross_series_dependence": 1.0426534821,
            "covariate_impulse_response": 0.5540589657
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.6010097051,
          "suiteCount": 1
        },
        {
          "model": "toto2.0",
          "scores": {
            "trend": 0.2431896043,
            "multi_seasonal": 0.6367942715,
            "time_varying_seasonality": 0.5711975158,
            "regime_switching": 0.144387832,
            "predictable_intermittency": 0.9554846888,
            "common_factor": 0.2906639314,
            "cross_series_dependence": 1.4597814138,
            "covariate_impulse_response": 0.6319093916
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.6166760812,
          "suiteCount": 1
        }
      ]
    },
    "GIFT-Medium": {
      "probe_mase": [
        {
          "model": "Chronos-2",
          "scores": {
            "trend": 2.9821687142,
            "multi_seasonal": 5.3094340502,
            "time_varying_seasonality": 7.0512128062,
            "regime_switching": 2.3783894114,
            "predictable_intermittency": 2.8511374044,
            "common_factor": 0.9187028112,
            "cross_series_dependence": 7.6898091022,
            "covariate_impulse_response": 5.1508306036
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 4.2914606129,
          "suiteCount": 1
        },
        {
          "model": "timesfm2.5",
          "scores": {
            "trend": 2.8804906278,
            "multi_seasonal": 4.0163978228,
            "time_varying_seasonality": 3.5069604366,
            "regime_switching": 2.4525856396,
            "predictable_intermittency": 2.936244793,
            "common_factor": 0.9658362122,
            "cross_series_dependence": 8.0885761794,
            "covariate_impulse_response": 6.8491753486
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 3.9620333825,
          "suiteCount": 1
        },
        {
          "model": "moirai2",
          "scores": {
            "trend": 3.4685549254,
            "multi_seasonal": 10.67080929,
            "time_varying_seasonality": 10.6157703202,
            "regime_switching": 4.0255619696,
            "predictable_intermittency": 3.5919038604,
            "common_factor": 1.3240486568,
            "cross_series_dependence": 9.9022135296,
            "covariate_impulse_response": 9.7734499948
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 6.6715390683,
          "suiteCount": 1
        },
        {
          "model": "Timer-3.5",
          "scores": {
            "trend": 2.807565928,
            "multi_seasonal": 5.5326933984,
            "time_varying_seasonality": 7.2918667858,
            "regime_switching": 2.5809815718,
            "predictable_intermittency": 3.0393835718,
            "common_factor": 0.9518983433,
            "cross_series_dependence": 8.0414163658,
            "covariate_impulse_response": 6.8482659262
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 4.6367589864,
          "suiteCount": 1
        },
        {
          "model": "toto2.0",
          "scores": {
            "trend": 12.785315754,
            "multi_seasonal": 10.4880783654,
            "time_varying_seasonality": 9.9546984936,
            "regime_switching": 7.596805884,
            "predictable_intermittency": 6.458565704,
            "common_factor": 1.1626623198,
            "cross_series_dependence": 15.988372016,
            "covariate_impulse_response": 19.0718701
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 10.4382960796,
          "suiteCount": 1
        }
      ],
      "paired_nrmse": [
        {
          "model": "Chronos-2",
          "scores": {
            "trend": 0.0510702675,
            "multi_seasonal": 1.1017202245,
            "time_varying_seasonality": 0.7866182767,
            "regime_switching": 0.0532803791,
            "predictable_intermittency": 1.4023369792,
            "common_factor": 0.2568179967,
            "cross_series_dependence": 0.9922667515,
            "covariate_impulse_response": 0.533018195
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.6471411338,
          "suiteCount": 1
        },
        {
          "model": "timesfm2.5",
          "scores": {
            "trend": 0.2592673928,
            "multi_seasonal": 0.9607989634,
            "time_varying_seasonality": 0.9795761154,
            "regime_switching": 0.1305952409,
            "predictable_intermittency": 1.321852674,
            "common_factor": 0.5244791105,
            "cross_series_dependence": 1.0530051996,
            "covariate_impulse_response": 0.3726284925
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.7002753986,
          "suiteCount": 1
        },
        {
          "model": "moirai2",
          "scores": {
            "trend": 0.2933982087,
            "multi_seasonal": 1.7289028614,
            "time_varying_seasonality": 1.5647137918,
            "regime_switching": 0.2761034234,
            "predictable_intermittency": 1.1152356236,
            "common_factor": 0.5709222302,
            "cross_series_dependence": 1.0506391562,
            "covariate_impulse_response": 0.4573683056
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.8821604501,
          "suiteCount": 1
        },
        {
          "model": "Timer-3.5",
          "scores": {
            "trend": 0.2637150076,
            "multi_seasonal": 1.2063428764,
            "time_varying_seasonality": 1.1295694081,
            "regime_switching": 0.19087616,
            "predictable_intermittency": 1.7384594048,
            "common_factor": 0.473736308,
            "cross_series_dependence": 1.0701185088,
            "covariate_impulse_response": 0.4975378205
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.8212944368,
          "suiteCount": 1
        },
        {
          "model": "toto2.0",
          "scores": {
            "trend": 0.590497429,
            "multi_seasonal": 1.1309571846,
            "time_varying_seasonality": 1.2959876571,
            "regime_switching": 0.175896607,
            "predictable_intermittency": 1.5562772876,
            "common_factor": 0.3771920467,
            "cross_series_dependence": 1.188800212,
            "covariate_impulse_response": 0.54968864
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.858162133,
          "suiteCount": 1
        }
      ]
    },
    "GIFT-Long": {
      "probe_mase": [
        {
          "model": "Chronos-2",
          "scores": {
            "trend": 3.5968965244,
            "multi_seasonal": 6.2523516718,
            "time_varying_seasonality": 4.8598837188,
            "regime_switching": 2.9341487456,
            "predictable_intermittency": 3.1306710602,
            "common_factor": 1.0392596282,
            "cross_series_dependence": 6.6688019332,
            "covariate_impulse_response": 9.909024956
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 4.7988797798,
          "suiteCount": 1
        },
        {
          "model": "timesfm2.5",
          "scores": {
            "trend": 3.681873234,
            "multi_seasonal": 4.8899137846,
            "time_varying_seasonality": 5.6321052564,
            "regime_switching": 3.253290489,
            "predictable_intermittency": 3.1854705764,
            "common_factor": 1.1132222228,
            "cross_series_dependence": 7.4459944046,
            "covariate_impulse_response": 9.5123886216
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 4.8392823237,
          "suiteCount": 1
        },
        {
          "model": "moirai2",
          "scores": {
            "trend": 4.6087321358,
            "multi_seasonal": 9.5424789602,
            "time_varying_seasonality": 9.7166923492,
            "regime_switching": 4.44188103,
            "predictable_intermittency": 4.6357462374,
            "common_factor": 1.7485751096,
            "cross_series_dependence": 10.128000262,
            "covariate_impulse_response": 14.300345048
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 7.3903063915,
          "suiteCount": 1
        },
        {
          "model": "Timer-3.5",
          "scores": {
            "trend": 3.5490162594,
            "multi_seasonal": 6.7795572582,
            "time_varying_seasonality": 8.0206724704,
            "regime_switching": 3.0999326214,
            "predictable_intermittency": 3.0874379826,
            "common_factor": 1.1441314376,
            "cross_series_dependence": 7.0046389768,
            "covariate_impulse_response": 13.4587322928
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 5.7680149124,
          "suiteCount": 1
        },
        {
          "model": "toto2.0",
          "scores": {
            "trend": 16.115202076,
            "multi_seasonal": 21.868927572,
            "time_varying_seasonality": 23.431939148,
            "regime_switching": 16.9942066662,
            "predictable_intermittency": 16.407114392,
            "common_factor": 1.323830949,
            "cross_series_dependence": 36.760570182,
            "covariate_impulse_response": 53.23788292
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 23.2674592381,
          "suiteCount": 1
        }
      ],
      "paired_nrmse": [
        {
          "model": "Chronos-2",
          "scores": {
            "trend": 0.0622699148,
            "multi_seasonal": 0.9144960364,
            "time_varying_seasonality": 0.8592633699,
            "regime_switching": 0.0644797099,
            "predictable_intermittency": 1.0510773702,
            "common_factor": 0.3851945981,
            "cross_series_dependence": 0.8932563752,
            "covariate_impulse_response": 0.3974530593
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.5784363042,
          "suiteCount": 1
        },
        {
          "model": "timesfm2.5",
          "scores": {
            "trend": 0.124415789,
            "multi_seasonal": 0.965956275,
            "time_varying_seasonality": 0.9610404233,
            "regime_switching": 0.1055890894,
            "predictable_intermittency": 1.0685443206,
            "common_factor": 0.7718095496,
            "cross_series_dependence": 1.0580044208,
            "covariate_impulse_response": 0.3351343653
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.6738117791,
          "suiteCount": 1
        },
        {
          "model": "moirai2",
          "scores": {
            "trend": 0.2132425505,
            "multi_seasonal": 1.6375312128,
            "time_varying_seasonality": 1.1899580104,
            "regime_switching": 0.3262622664,
            "predictable_intermittency": 1.6831258096,
            "common_factor": 1.0842361151,
            "cross_series_dependence": 1.0186347863,
            "covariate_impulse_response": 0.4903511629
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.9554177392,
          "suiteCount": 1
        },
        {
          "model": "Timer-3.5",
          "scores": {
            "trend": 0.1909827222,
            "multi_seasonal": 1.167812185,
            "time_varying_seasonality": 1.2226567763,
            "regime_switching": 0.174551316,
            "predictable_intermittency": 1.1182987154,
            "common_factor": 0.7386587285,
            "cross_series_dependence": 0.9945917009,
            "covariate_impulse_response": 0.5936593505
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.7751514368,
          "suiteCount": 1
        },
        {
          "model": "toto2.0",
          "scores": {
            "trend": 0.2572961598,
            "multi_seasonal": 0.9977196967,
            "time_varying_seasonality": 1.1187312314,
            "regime_switching": 0.2649193968,
            "predictable_intermittency": 1.1840213712,
            "common_factor": 0.66308056,
            "cross_series_dependence": 1.3256110726,
            "covariate_impulse_response": 0.8574889064
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.8336085494,
          "suiteCount": 1
        }
      ]
    },
    "FEV-Mini20": {
      "probe_mase": [
        {
          "model": "Chronos-2",
          "scores": {
            "trend": 1.9384934664,
            "multi_seasonal": 2.655597812,
            "time_varying_seasonality": 2.1896014556,
            "regime_switching": 1.912631364,
            "predictable_intermittency": 2.1878520142,
            "common_factor": 1.7662260486,
            "cross_series_dependence": 1.7661408328,
            "covariate_impulse_response": 3.073196168
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 2.1862173952,
          "suiteCount": 1
        },
        {
          "model": "timesfm2.5",
          "scores": {
            "trend": 1.4688477124,
            "multi_seasonal": 2.5817526418,
            "time_varying_seasonality": 1.7438264198,
            "regime_switching": 1.561859793,
            "predictable_intermittency": 1.7117031018,
            "common_factor": 1.629707924,
            "cross_series_dependence": 1.6777015876,
            "covariate_impulse_response": 2.4795290768
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 1.8568660322,
          "suiteCount": 1
        },
        {
          "model": "tirex2",
          "scores": {
            "trend": 1.736174316,
            "multi_seasonal": 2.7030957802,
            "time_varying_seasonality": 2.0404532228,
            "regime_switching": 1.853036877,
            "predictable_intermittency": 2.1537930404,
            "common_factor": 1.8193962604,
            "cross_series_dependence": 1.6934071802,
            "covariate_impulse_response": 2.9717750592
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 2.121391467,
          "suiteCount": 1
        },
        {
          "model": "moirai2",
          "scores": {
            "trend": 1.6249590262,
            "multi_seasonal": 3.1443787084,
            "time_varying_seasonality": 2.0453971838,
            "regime_switching": 1.6470043594,
            "predictable_intermittency": 2.0551494928,
            "common_factor": 1.7912594182,
            "cross_series_dependence": 1.711298697,
            "covariate_impulse_response": 3.024641324
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 2.1305110262,
          "suiteCount": 1
        },
        {
          "model": "Timer-3.5",
          "scores": {
            "trend": 1.8074029298,
            "multi_seasonal": 3.003238014,
            "time_varying_seasonality": 2.1396020992,
            "regime_switching": 1.917040952,
            "predictable_intermittency": 1.987160354,
            "common_factor": 2.0435737818,
            "cross_series_dependence": 2.1002584676,
            "covariate_impulse_response": 3.0464334846
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 2.2555887604,
          "suiteCount": 1
        },
        {
          "model": "toto2.0",
          "scores": {
            "trend": 1.5906668714,
            "multi_seasonal": 1.8393547824,
            "time_varying_seasonality": 1.6592905032,
            "regime_switching": 1.6030404572,
            "predictable_intermittency": 1.7309097292,
            "common_factor": 1.6626594216,
            "cross_series_dependence": 1.8258598076,
            "covariate_impulse_response": 2.302633004
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 1.7768018221,
          "suiteCount": 1
        }
      ],
      "paired_nrmse": [
        {
          "model": "Chronos-2",
          "scores": {
            "trend": 0.1764511114,
            "multi_seasonal": 1.201830313,
            "time_varying_seasonality": 0.8119565624,
            "regime_switching": 0.2125384652,
            "predictable_intermittency": 0.9338045567,
            "common_factor": 0.2861930493,
            "cross_series_dependence": 0.7181094174,
            "covariate_impulse_response": 0.7763432392
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.6396533393,
          "suiteCount": 1
        },
        {
          "model": "timesfm2.5",
          "scores": {
            "trend": 0.1112489369,
            "multi_seasonal": 0.9536108276,
            "time_varying_seasonality": 0.7335717233,
            "regime_switching": 0.1197441879,
            "predictable_intermittency": 0.8644769746,
            "common_factor": 0.3011025411,
            "cross_series_dependence": 0.7629443771,
            "covariate_impulse_response": 0.6906901586
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.5671737159,
          "suiteCount": 1
        },
        {
          "model": "tirex2",
          "scores": {
            "trend": 0.1302019242,
            "multi_seasonal": 1.0422065275,
            "time_varying_seasonality": 0.7938957873,
            "regime_switching": 0.1362862374,
            "predictable_intermittency": 0.8940457383,
            "common_factor": 0.2910784938,
            "cross_series_dependence": 0.7172170315,
            "covariate_impulse_response": 0.6506887495
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.5819525612,
          "suiteCount": 1
        },
        {
          "model": "moirai2",
          "scores": {
            "trend": 0.0874939519,
            "multi_seasonal": 1.2125730872,
            "time_varying_seasonality": 0.7974517024,
            "regime_switching": 0.1026859614,
            "predictable_intermittency": 1.0622240653,
            "common_factor": 0.3802409627,
            "cross_series_dependence": 0.8342275578,
            "covariate_impulse_response": 0.6744868031
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.6439230115,
          "suiteCount": 1
        },
        {
          "model": "Timer-3.5",
          "scores": {
            "trend": 0.1807181212,
            "multi_seasonal": 1.1429272928,
            "time_varying_seasonality": 0.7555655496,
            "regime_switching": 0.1632377905,
            "predictable_intermittency": 0.9629794435,
            "common_factor": 0.3589430479,
            "cross_series_dependence": 0.7818999719,
            "covariate_impulse_response": 0.6419192902
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.6235238134,
          "suiteCount": 1
        },
        {
          "model": "toto2.0",
          "scores": {
            "trend": 0.1235699418,
            "multi_seasonal": 0.8093271822,
            "time_varying_seasonality": 0.6626374434,
            "regime_switching": 0.1100236796,
            "predictable_intermittency": 0.8516150683,
            "common_factor": 0.2951707795,
            "cross_series_dependence": 0.7517081094,
            "covariate_impulse_response": 0.5506950119
          },
          "coverage": {
            "trend": 1,
            "multi_seasonal": 1,
            "time_varying_seasonality": 1,
            "regime_switching": 1,
            "predictable_intermittency": 1,
            "common_factor": 1,
            "cross_series_dependence": 1,
            "covariate_impulse_response": 1
          },
          "average": 0.519343402,
          "suiteCount": 1
        }
      ]
    },
    "all": {
      "probe_mase": [
        {
          "model": "Chronos-2",
          "scores": {
            "trend": 2.6624197357,
            "multi_seasonal": 4.1467841955,
            "time_varying_seasonality": 3.90659921,
            "regime_switching": 2.2320024422,
            "predictable_intermittency": 2.7746439558,
            "common_factor": 1.0854399596,
            "cross_series_dependence": 4.7550621218,
            "covariate_impulse_response": 5.4428716116
          },
          "coverage": {
            "trend": 4,
            "multi_seasonal": 4,
            "time_varying_seasonality": 4,
            "regime_switching": 4,
            "predictable_intermittency": 4,
            "common_factor": 4,
            "cross_series_dependence": 4,
            "covariate_impulse_response": 4
          },
          "average": 3.375727904,
          "suiteCount": 4
        },
        {
          "model": "timesfm2.5",
          "scores": {
            "trend": 2.4823182483,
            "multi_seasonal": 3.4968298351,
            "time_varying_seasonality": 3.0863051787,
            "regime_switching": 2.2236330672,
            "predictable_intermittency": 2.6665150578,
            "common_factor": 1.0892098855,
            "cross_series_dependence": 5.0270842542,
            "covariate_impulse_response": 5.5829090608
          },
          "coverage": {
            "trend": 4,
            "multi_seasonal": 4,
            "time_varying_seasonality": 4,
            "regime_switching": 4,
            "predictable_intermittency": 4,
            "common_factor": 4,
            "cross_series_dependence": 4,
            "covariate_impulse_response": 4
          },
          "average": 3.2068505735,
          "suiteCount": 4
        },
        {
          "model": "tirex2",
          "scores": {
            "trend": 1.7895930901,
            "multi_seasonal": 2.6174306406,
            "time_varying_seasonality": 1.7620650593,
            "regime_switching": 1.7252435659,
            "predictable_intermittency": 2.4496181562,
            "common_factor": 1.238513283,
            "cross_series_dependence": 2.2791349608,
            "covariate_impulse_response": 3.2281573419
          },
          "coverage": {
            "trend": 2,
            "multi_seasonal": 2,
            "time_varying_seasonality": 2,
            "regime_switching": 2,
            "predictable_intermittency": 2,
            "common_factor": 2,
            "cross_series_dependence": 2,
            "covariate_impulse_response": 2
          },
          "average": 2.1362195122,
          "suiteCount": 4
        },
        {
          "model": "moirai2",
          "scores": {
            "trend": 2.9635308436,
            "multi_seasonal": 6.8822500021,
            "time_varying_seasonality": 6.0577609891,
            "regime_switching": 3.0060821837,
            "predictable_intermittency": 3.6235290965,
            "common_factor": 1.389720599,
            "cross_series_dependence": 6.2003606112,
            "covariate_impulse_response": 8.2514105342
          },
          "coverage": {
            "trend": 4,
            "multi_seasonal": 4,
            "time_varying_seasonality": 4,
            "regime_switching": 4,
            "predictable_intermittency": 4,
            "common_factor": 4,
            "cross_series_dependence": 4,
            "covariate_impulse_response": 4
          },
          "average": 4.7968306074,
          "suiteCount": 4
        },
        {
          "model": "Timer-3.5",
          "scores": {
            "trend": 2.5326008183,
            "multi_seasonal": 4.5184436849,
            "time_varying_seasonality": 4.7976142557,
            "regime_switching": 2.3287121077,
            "predictable_intermittency": 2.7464014102,
            "common_factor": 1.1958383433,
            "cross_series_dependence": 5.0320289931,
            "covariate_impulse_response": 6.7966812335
          },
          "coverage": {
            "trend": 4,
            "multi_seasonal": 4,
            "time_varying_seasonality": 4,
            "regime_switching": 4,
            "predictable_intermittency": 4,
            "common_factor": 4,
            "cross_series_dependence": 4,
            "covariate_impulse_response": 4
          },
          "average": 3.7435401058,
          "suiteCount": 4
        },
        {
          "model": "toto2.0",
          "scores": {
            "trend": 8.0918966777,
            "multi_seasonal": 9.0326249793,
            "time_varying_seasonality": 9.1544755117,
            "regime_switching": 6.941922348,
            "predictable_intermittency": 6.8238113814,
            "common_factor": 1.1935090681,
            "cross_series_dependence": 14.3302052054,
            "covariate_impulse_response": 19.5336026018
          },
          "coverage": {
            "trend": 4,
            "multi_seasonal": 4,
            "time_varying_seasonality": 4,
            "regime_switching": 4,
            "predictable_intermittency": 4,
            "common_factor": 4,
            "cross_series_dependence": 4,
            "covariate_impulse_response": 4
          },
          "average": 9.3877559717,
          "suiteCount": 4
        }
      ],
      "paired_nrmse": [
        {
          "model": "Chronos-2",
          "scores": {
            "trend": 0.0929684508,
            "multi_seasonal": 0.9993744825,
            "time_varying_seasonality": 0.7468290959,
            "regime_switching": 0.1066709103,
            "predictable_intermittency": 1.0730900728,
            "common_factor": 0.2962677775,
            "cross_series_dependence": 0.8982276248,
            "covariate_impulse_response": 0.583226716
          },
          "coverage": {
            "trend": 4,
            "multi_seasonal": 4,
            "time_varying_seasonality": 4,
            "regime_switching": 4,
            "predictable_intermittency": 4,
            "common_factor": 4,
            "cross_series_dependence": 4,
            "covariate_impulse_response": 4
          },
          "average": 0.5995818913,
          "suiteCount": 4
        },
        {
          "model": "timesfm2.5",
          "scores": {
            "trend": 0.1776293275,
            "multi_seasonal": 0.9205062881,
            "time_varying_seasonality": 0.8384013658,
            "regime_switching": 0.1117208179,
            "predictable_intermittency": 1.0230567584,
            "common_factor": 0.4826772461,
            "cross_series_dependence": 1.0151617053,
            "covariate_impulse_response": 0.4733425406
          },
          "coverage": {
            "trend": 4,
            "multi_seasonal": 4,
            "time_varying_seasonality": 4,
            "regime_switching": 4,
            "predictable_intermittency": 4,
            "common_factor": 4,
            "cross_series_dependence": 4,
            "covariate_impulse_response": 4
          },
          "average": 0.6303120062,
          "suiteCount": 4
        },
        {
          "model": "tirex2",
          "scores": {
            "trend": 0.1932192071,
            "multi_seasonal": 0.9431549088,
            "time_varying_seasonality": 0.687094528,
            "regime_switching": 0.1035589843,
            "predictable_intermittency": 0.933267388,
            "common_factor": 0.2748933376,
            "cross_series_dependence": 0.8608962252,
            "covariate_impulse_response": 0.7504966546
          },
          "coverage": {
            "trend": 2,
            "multi_seasonal": 2,
            "time_varying_seasonality": 2,
            "regime_switching": 2,
            "predictable_intermittency": 2,
            "common_factor": 2,
            "cross_series_dependence": 2,
            "covariate_impulse_response": 2
          },
          "average": 0.5933226542,
          "suiteCount": 4
        },
        {
          "model": "moirai2",
          "scores": {
            "trend": 0.2022734034,
            "multi_seasonal": 1.3972908063,
            "time_varying_seasonality": 1.0731518683,
            "regime_switching": 0.2098029093,
            "predictable_intermittency": 1.206477635,
            "common_factor": 0.5971660888,
            "cross_series_dependence": 0.9868646214,
            "covariate_impulse_response": 0.5766529772
          },
          "coverage": {
            "trend": 4,
            "multi_seasonal": 4,
            "time_varying_seasonality": 4,
            "regime_switching": 4,
            "predictable_intermittency": 4,
            "common_factor": 4,
            "cross_series_dependence": 4,
            "covariate_impulse_response": 4
          },
          "average": 0.7812100387,
          "suiteCount": 4
        },
        {
          "model": "Timer-3.5",
          "scores": {
            "trend": 0.2196189928,
            "multi_seasonal": 1.1156940084,
            "time_varying_seasonality": 0.945530951,
            "regime_switching": 0.1654005999,
            "predictable_intermittency": 1.1835364401,
            "common_factor": 0.4680680196,
            "cross_series_dependence": 0.9723159159,
            "covariate_impulse_response": 0.5717938567
          },
          "coverage": {
            "trend": 4,
            "multi_seasonal": 4,
            "time_varying_seasonality": 4,
            "regime_switching": 4,
            "predictable_intermittency": 4,
            "common_factor": 4,
            "cross_series_dependence": 4,
            "covariate_impulse_response": 4
          },
          "average": 0.705244848,
          "suiteCount": 4
        },
        {
          "model": "toto2.0",
          "scores": {
            "trend": 0.3036382837,
            "multi_seasonal": 0.8936995838,
            "time_varying_seasonality": 0.9121384619,
            "regime_switching": 0.1738068788,
            "predictable_intermittency": 1.136849604,
            "common_factor": 0.4065268294,
            "cross_series_dependence": 1.181475202,
            "covariate_impulse_response": 0.6474454875
          },
          "coverage": {
            "trend": 4,
            "multi_seasonal": 4,
            "time_varying_seasonality": 4,
            "regime_switching": 4,
            "predictable_intermittency": 4,
            "common_factor": 4,
            "cross_series_dependence": 4,
            "covariate_impulse_response": 4
          },
          "average": 0.7069475414,
          "suiteCount": 4
        }
      ]
    }
  }
};
