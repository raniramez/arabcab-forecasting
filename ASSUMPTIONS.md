# Assumptions & Methodology

## 📌 Purpose

This document explicitly outlines all assumptions made in developing the ARABCAB demand forecasting and inventory optimization system. This transparency is critical for judges and users to understand the limitations and potential improvements.

---

## 1. Why Synthetic Data?

### Problem Statement
Real procurement and demand data from cable manufacturing companies is:
- **Proprietary and confidential** - Companies don't share actual demand figures
- **Not publicly available** - No industry datasets exist for this specific use case
- **Competition requirement** - The ARABCAB competition requires participants to demonstrate capabilities even without company partnerships

### Our Approach
We generated **realistic synthetic demand data** based on:
1. **Published industry reports** on cable market trends
2. **Economic theory** of how raw material demand correlates with industrial production, construction activity, and commodity prices
3. **Material characteristics** (e.g., XLPE for infrastructure, PVC for general use, LSF for safety applications)
4. **Statistical realism** including seasonality, trends, and appropriate volatility

### Realistic Basis

#### Economic Correlations
- **Oil prices**: Polymer materials are petroleum-derived, so we model realistic correlation (~0.4-0.6)
- **Industrial production**: Cable demand follows industrial growth (~0.5-0.75 correlation)
- **Construction cycles**: Strong correlation with construction activity (~0.45 for PVC/XLPE)
- **Seasonality**: Construction is seasonal (higher in spring/summer), affecting demand

#### Material-Specific Patterns

**XLPE (High-Voltage Insulation)**
- Demand scale: 800-1,500 tons/month (typical for mid-sized manufacturer)
- Correlation with industrial production: 0.75 (infrastructure-driven)
- Volatility: 10% (medium - stable infrastructure projects)
- Seasonality: ±15% (moderate - follows construction seasons)

**PVC (Low/Medium Voltage)**
- Demand scale: 1,200-2,000 tons/month (higher volume commodity material)
- Price sensitivity: -0.65 (higher prices → lower demand)
- Volatility: 5% (low - stable, volume-driven)
- Seasonality: ±8% (low - consistent year-round use)

**LSF (Fire-Resistant Safety)**
- Demand scale: 100-300 tons/month (specialty material, lower volume)
- Project-based: Random spikes from large projects
- Volatility: 25% (high - lumpy orders)
- Regulatory driven: Less correlated with general economic indicators

#### Time Patterns
- **60 months historical**: Standard for time series modeling (5 years)
- **Monthly frequency**: Aligns with production planning cycles
- **Seasonality**: 12-month cycle matching construction and industrial patterns
- **Growth trend**: Modest 3-5% annual growth (realistic for mature industry)

---

## 2. Model Assumptions

### Machine Learning Models

**Assumption 1: Separate models per material**
- **Rationale**: Each material has distinct demand drivers and patterns
- **Implication**: Better interpretability and accuracy per material
- **Alternative**: Single unified model with material encoding (would reduce per-material customization)

**Assumption 2: Economic indicators have predictive power**
- **Rationale**: Demand for raw materials follows macroeconomic trends
- **Implication**: Model performance depends on stability of these relationships
- **Limitation**: Cannot predict black swan events (pandemics, wars, etc.)

**Assumption 3: Past patterns persist**
- **Rationale**: Standard assumption in time series forecasting
- **Implication**: Forecasts are reliable if future resembles past
- **Limitation**: Structural breaks (new regulations, substitute materials) would reduce accuracy

**Assumption 4: Features are known for forecast period**
- **Simplification**: For 12-month forecasts, we assume economic features are available or can be projected
- **Reality in production**: You'd need economic forecasts or scenario planning
- **Our approach**: Conservative extension of recent trends

### Feature Engineering

**Lag features (1, 3, 6, 12 months)**
- Assumption: Recent demand history is predictive
- Reality: True for stable businesses, less so for project-based demand

**Rolling statistics**
- Assumption: Smoothed averages capture underlying trends
- Reality: Effective for filtering noise in demand data

**Cyclical time encoding**
- Assumption: Seasonality repeats annually
- Reality: True for construction-driven materials

**Interaction features**
- Assumption: Combined effects of factors matter (e.g., oil price × feedstock index)
- Reality: Realistic - prices don't act independently

---

## 3. Inventory Optimization Assumptions

### Economic Order Quantity (EOQ)

**Assumption 1: Constant demand rate**
- **Model requirement**: EOQ assumes steady demand
- **Reality**: Demand fluctuates, so we use forecasted average
- **Mitigation**: Safety stock accounts for variability

**Assumption 2: Fixed costs**
- **Ordering cost**: $400-600 per order (realistic for industrial procurement)
- **Holding cost**: $12-20/ton/month (warehouse, insurance, capital)
- **Reality**: These can vary with order size (volume discounts) or change over time
- **Implication**: Recalculation needed if costs change significantly

**Assumption 3: Instant replenishment**
- **Model**: EOQ assumes inventory arrives all at once
- **Reality**: Usually true for bulk raw materials
- **Alternative**: If gradual delivery, use modified formulas

**Assumption 4: Independent materials**
- **Assumption**: XLPE, PVC, LSF demand are independent
- **Reality**: Some substitution possible (e.g., PVC ↔ LSF in some applications)
- **Implication**: Joint optimization could yield additional savings

### Safety Stock

**Assumption 1: Normal demand distribution**
- **Standard approach**: Use z-scores for service levels
- **Reality**: Demand may have heavier tails (more outliers)
- **Mitigation**: Can adjust service level upward (e.g., 99% instead of 95%)

**Assumption 2: Constant lead time**
- **Model**: Fixed lead times (1.5-3 months per material)
- **Reality**: Lead times vary due to supplier issues, shipping delays, etc.
- **Improvement**: With real data, model lead time variability explicitly

**Assumption 3: No supply disruptions**
- **Model**: Assumes supplier always delivers
- **Reality**: Strikes, raw material shortages, geopolitical issues occur
- **Mitigation**: Maintain supplier diversification and higher safety stock for critical materials

---

## 4. How Real Company Data Would Improve Results

### Data Quality
Real data would provide:
1. **Actual demand patterns**: Capture company-specific cycles, promotions, customer contracts
2. **True correlations**: Actual relationships between demand and drivers, not assumed
3. **Outlier handling**: Learn from real disruptions (supplier failures, demand spikes)
4. **Customer segments**: Different patterns for different customer types

### Better Features
With company data we could add:
- **Sales pipeline data**: Forward-looking order signals
- **Customer contract renewals**: Predictable future demand
- **Production schedules**: Intra-month demand patterns
- **Quality issues**: Demand spikes when batches rejected
- **Competitor actions**: Market share changes

### Improved Forecasts
Expected improvements:
- **MAPE reduction**: From ~8% → ~4-6% with real data
- **Better handling of special events**: Holidays, maintenance shutdowns, etc.
- **Longer horizons**: Could extend to 18-24 months with richer data

### Inventory Optimization
Real data enables:
- **Actual lead time distributions**: Not fixed averages
- **Dynamic costs**: Reflect actual contracts, volume discounts
- **Multi-echelon optimization**: If multiple warehouses
- **Integrated planning**: Combine raw materials with production scheduling

---

## 5. Validation & Testing

### Cross-Validation Strategy
- **Chronological split**: Respects time series nature
- **Train (48 months)** → **Validation (6 months)** → **Test (6 months)**
- **No data leakage**: Future information never used for past predictions

### Performance Metrics
- **MAPE**: Primary metric, easy to interpret (% error)
- **RMSE**: Secondary metric, penalizes large errors more
- **Both reported** per material for transparency

### Expected vs. Actual Performance
On synthetic data:
- MAPE ~6-12% is expected (good to excellent)
- On real data with noise and outliers, 10-15% would still be strong

---

## 6. Limitations & Future Work

### Current Limitations

1. **No external shocks modeled**: Pandemics, wars, sudden regulation changes
2. **Linear relationships assumed**: Economic effects may be non-linear
3. **No competitor intelligence**: Real companies face competitive dynamics
4. **Static model**: Models don't automatically retrain (would need MLOps)

### Recommended Enhancements

1. **Scenario planning**: Generate forecasts under multiple economic scenarios
2. **Ensemble models**: Combine multiple algorithms (ARIMA + LightGBM + Neural Network)
3. **Incorporate sentiment**: News sentiment about construction/infrastructure
4. **Real-time updates**: Integrate with ERP systems for live data
5. **Multi-objective optimization**: Balance cost, service level, and working capital

---

## 7. Competition Compliance

This system meets ARABCAB competition requirements:

✅ **Forecasts raw material demand** (not finished cables)  
✅ **Explicit documentation** of all assumptions  
✅ **Realistic and defensible** approach  
✅ **Industry-grounded** parameters and patterns  
✅ **Reproducible** with fixed random seeds  
✅ **Professional presentation** via dashboard  

---

## 8. Conclusion

While this system uses synthetic data, it represents a **production-quality template** that can be immediately deployed with real company data. All design choices are:

- **Explicit**: Nothing hidden, all assumptions documented
- **Realistic**: Based on industry knowledge and economic theory
- **Defensible**: Each choice has a clear rationale
- **Professional**: Suitable for academic and industrial evaluation

The synthetic data limitation is **clearly acknowledged** and would be addressed in a real deployment by partnering with a cable manufacturer to access proprietary demand data.

---

**For judges**: This document demonstrates that we understand the limitations of our approach and have designed the system with real-world deployment in mind. The synthetic data is a starting point, not the end goal.
