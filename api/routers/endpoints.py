from fastapi import APIRouter, HTTPException
from ml_core.analyzer import ResearchAnalyzer
from ..schemas import AnalysisRequest, CompositeRequest, SubsetRequest, AnalysisResponse
import pandas as pd

router = APIRouter(prefix="/api/analyze")

# Один экземпляр анализатора на всё приложение
analyzer = ResearchAnalyzer()


@router.post("/full", response_model=AnalysisResponse)
async def full_analysis(request: AnalysisRequest):
    try:
        df = pd.DataFrame(request.df)

        result = analyzer.run_full_analysis(
            df=df,
            target_col=request.target_col,
            n_clusters=request.n_clusters,
            corr_threshold=request.corr_threshold,
            use_smote=request.use_smote
        )

        return AnalysisResponse(
            metrics=result.get("metrics", {}),
            selected_features=result.get("selected_features", []),
            cluster_profiles=result.get("cluster_profiles", {}),
            explanations=result.get("explanations", []),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")


@router.post("/composite/create")
async def create_composite(request: CompositeRequest):
    try:
        df = pd.DataFrame(request.df)
        df_new, score_name = analyzer.create_composite_score(
            df, request.feature_weights, request.score_name
        )

        return {
            "score_name": score_name,
            "statistics": df_new[score_name].describe().to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subset/select")
async def select_subset(request: SubsetRequest):
    try:
        df = pd.DataFrame(request.df)
        subset = analyzer.select_subset(
            df,
            condition=request.condition,
            n_samples=request.n_samples,
            by_cluster=request.by_cluster
        )
        return {
            "count": len(subset),
            "data": subset.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))