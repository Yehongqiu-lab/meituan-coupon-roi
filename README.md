# meituan-coupon-roi
Improve Marketing ROI in coupons: An end-to-end data science product for a e-commerce platform

**Project Introduction 项目来源和商业问题**
The project is from the 2025 Business Intelligence Competition held by Gou Xiong Hui and Meituan. Meituan is a China-based O2O e-commerce platform. The project data is ~1G+ Meituan de-sensitized user daily activity data from Jan. 2023 to Jul. 2023.
The platform distributed over `13M` coupons within the winter-spring seasons in 2023, while only `600K` were redeemed and converted to actual profits, with the redemption rate `<5%`. The key question is how to improve the coupon redemption rate, as well as the margin each coupon brings in to the platform.

项目来源为2025 年狗熊会大学生商业分析大赛；数据为 1G+ 美团平台 2023 上半年用户日活数据。平台半年累计发放近 13M 优惠券，仅 600K 被领取使用，转化率不足 5%。海量发放优惠券不仅低效，还降低用户体验。

**Research Method and Goal 研究方法和项目目标**
The project aims to train machine learning models to predict the odds of each user redeeming differet types of coupons given their previous activity behaviours. With machine learning models, we can predict more accurately on the potential value of coupons and distribute individualized coupons.

基于机器学习模型的预测，个性化发放优惠券，提高投放精准度和整体效率。

**Project Process Management 过程管理**
The project uses Git for code version control. Model training pipelines are standardized for fast iteration of different configurations of the LightGBM and CatBoost models. Important functions in source codes are unit-tested with Pytest. The data ETL process is completed with the SQL API of DuckDB.

利用 Git 进行代码版本控制，标准化机器学习训练工作流，快速迭代 LightGBM 和 CatBoost 机器学习模型；使用 Pytest 对关键代码实现单元测试；用 SQL 完成数据 ETL。

**Business Value 商业价值**
By distributing only the coupons predicted as `>0.5` for their potential redemption rate, the margin of each coupon increases to `>2.5￥` from `1.3￥`, and the coupons with the highest 5% redemption rate prediction score striking an actual redemption rate of `>20%`.

优惠券平均转化利润由 1.3 元提升到 2.5+ 元，预测排名前 5% 的优惠券领取率高达 20%+。

## Intial BI Report
https://pickled-diploma-981.notion.site/An-EDA-into-the-conversion-funnel-datasets-of-an-e-commerce-platform-Meituan-1ff1629ab9d3800d9d10cc015e193bee?pvs=74

## ML Experimenting Log
https://pickled-diploma-981.notion.site/Model-Files-name-and-Model-performance-notes-2dd1629ab9d3807ebe58df5d46d936d6
