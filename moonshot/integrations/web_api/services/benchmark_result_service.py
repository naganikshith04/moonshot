from .... import api as moonshot_api
from ..services.base_service import BaseService
from ..services.utils.exceptions_handler import exception_handler
from ..types.types import BenchmarkResult


class BenchmarkResultService(BaseService):
    @exception_handler
    def get_all_results(self,type: str) -> list[BenchmarkResult]:
        if type == "cookbook":
            retn_list = []
            results=moonshot_api.api_get_all_result()
            for result in results:
                if result.get('metadata', {}).get('cookbooks') is None:
                    continue
                else:
                    retn_list.append(result)
            return [BenchmarkResult(**result) for result in retn_list]
        else:
            results = moonshot_api.api_get_all_result()
        return [BenchmarkResult(**result) for result in results]

    @exception_handler
    def get_all_result_name(self) -> list[str] | None:
        results = moonshot_api.api_get_all_result_name()
        return results

    @exception_handler
    def get_result_by_id(self, result_id: str) -> BenchmarkResult:
        result = moonshot_api.api_read_result(result_id)
        return BenchmarkResult(**result)

    @exception_handler
    def delete_result(self, result_id: str) -> None:
        moonshot_api.api_delete_result(result_id)
