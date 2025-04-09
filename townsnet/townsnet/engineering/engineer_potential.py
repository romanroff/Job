import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.geometry.base import BaseGeometry
from typing import Dict, Any, Set, List
from enum import Enum

class EngineeringObject(Enum):
    POWER_SUPPLY = 'Энергоснабжение'
    HEAT_SUPPLY = 'Теплоснабжение'
    GAS_SUPPLY = 'Газоснабжение'
    WATER_SUPPLY = 'Водоснабжение'
    WATER_DRAINAGE = 'Водоотведение'

class InfrastructureAnalyzer:
    def __init__(self, infrastructure_gdf: gpd.GeoDataFrame, assessment_areas_gdf: gpd.GeoDataFrame) -> None:
        # Преобразуем в метрическую систему координат (EPSG:3857)
        self.infrastructure_gdf = infrastructure_gdf.to_crs(epsg=3857)
        self.assessment_areas_gdf = assessment_areas_gdf.to_crs(epsg=3857)

        # Добавляем столбцы для результатов
        self.assessment_areas_gdf['score'] = 0
        self.assessment_areas_gdf['types_in_radius'] = None

        # Создаём пространственный индекс GeoPandas
        self.infrastructure_sindex = self.infrastructure_gdf.sindex

        # Начинаем анализ
        self._analyze_infrastructure()

    @staticmethod
    def get_radius(physical_object_type: Dict[str, Any]) -> float:
        """Определяет радиус на основе типа объекта."""
        name = physical_object_type.get('name', '')
        if "Атомная электростанция" in name:
            return 100000.0  # 100 км
        elif "Гидроэлектростанция" in name:
            return 10000.0   # 10 км
        else:
            return 1000.0    # 1 км
        
    @staticmethod
    def _get_description(types_in_radius : list[str]):
        all_types = {eng_obj.value for eng_obj in list(EngineeringObject)}
        existing_types = set(types_in_radius)
        non_existing_types = all_types - existing_types

        existing_types_str = str.join(', ', [t.lower() for t in existing_types])
        non_existing_types_str = str.join(', ', [t.lower() for t in non_existing_types])

        description = ''
        if len(existing_types) > 0 and len(non_existing_types) == 0:
            types_str = str.join(', ', [existing_types_str, non_existing_types_str])
            description = f'В радиусе доступности присутствуют все необходимые инженерные объекты: {types_str}.'
        elif len(existing_types) == 0 and len(non_existing_types) > 0:
            types_str = str.join(', ', [existing_types_str, non_existing_types_str])
            description = f'В радиусе отсутствуют необходимые инженерные объекты: {types_str}.'
        elif len(existing_types) > 0 and len(non_existing_types) > 0:
            description = f'В радиусе доступности присутствуют необходимые инженерные объекты: {existing_types_str}. Однако, отсутствуют другие объекты инженерной инфраструктуры: {non_existing_types_str}.'
        
        return description

    def _analyze_infrastructure(self) -> None:
        """Анализирует инфраструктуру для каждой области оценки."""
        # Предварительно вычисляем радиусы буферов для объектов инфраструктуры
        self.infrastructure_gdf['buffer_radius'] = self.infrastructure_gdf['physical_object_type'].apply(self.get_radius)

        # Основной цикл по зонам оценки
        for idx, area in self.assessment_areas_gdf.iterrows():
            area_geometry: BaseGeometry = area.geometry
            unique_types_in_radius: Set[str] = set()

            # Найти кандидатов с использованием пространственного индекса
            possible_matches_idx = list(self.infrastructure_sindex.intersection(area_geometry.bounds))
            possible_matches = self.infrastructure_gdf.iloc[possible_matches_idx]

            for _, obj in possible_matches.iterrows():
                obj_geometry = obj.geometry
                buffer_radius = obj['buffer_radius']

                # Проверка пересечения линии или попадания точки в буфер
                if isinstance(obj_geometry, LineString):
                    # Линии: проверяем пересечение линии с зоной оценки
                    if obj_geometry.intersects(area_geometry):
                        unique_types_in_radius.add(obj['type'])
                else:
                    # Для точек и других объектов: стандартная проверка
                    if obj_geometry.within(area_geometry.buffer(buffer_radius)):
                        unique_types_in_radius.add(obj['type'])

            # Сохраняем результаты
            self.assessment_areas_gdf.at[idx, 'score'] = len(unique_types_in_radius)
            self.assessment_areas_gdf.at[idx, 'types_in_radius'] = list(unique_types_in_radius)
            self.assessment_areas_gdf.at[idx, 'description'] = self._get_description(unique_types_in_radius)

    def get_results(self) -> gpd.GeoDataFrame:
        """Возвращает результат в CRS 4326."""
        return self.assessment_areas_gdf[['score', 'types_in_radius', 'description', 'geometry']].to_crs(epsg=4326)

