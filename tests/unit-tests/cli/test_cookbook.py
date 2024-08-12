from ast import literal_eval
from unittest.mock import AsyncMock, patch

import pytest

from moonshot.integrations.cli.benchmark.cookbook import (
    add_cookbook,
    list_cookbooks,
    run_cookbook,
    view_cookbook,
)


class TestCollectionCliCookbook:
    @pytest.fixture(autouse=True)
    def init(self):
        # Perform tests
        yield

    # ------------------------------------------------------------------------------
    # Test add_cookbook functionality
    # ------------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "name, description, recipes, expected_output",
        [
            # Valid case
            (
                "Test Cookbook",
                "This is a test cookbook.",
                "['recipe1', 'recipe2']",
                "[add_cookbook]: Cookbook (new_cookbook_id) created.",
            ),
            (
                "Another Cookbook",
                "Another description.",
                "['recipe3']",
                "[add_cookbook]: Cookbook (new_cookbook_id) created.",
            ),
            # Invalid case for name
            (
                None,
                "This is a test cookbook.",
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            (
                "",
                "This is a test cookbook.",
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            (
                99,
                "This is a test cookbook.",
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            (
                {},
                "This is a test cookbook.",
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            (
                [],
                "This is a test cookbook.",
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            (
                (),
                "This is a test cookbook.",
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            (
                True,
                "This is a test cookbook.",
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            (
                "",
                "This is a test cookbook.",
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            # Invalid case for description
            (
                "Test Cookbook",
                None,
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'description' argument must be a non-empty string and not None.",
            ),
            (
                "Test Cookbook",
                "",
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'description' argument must be a non-empty string and not None.",
            ),
            (
                "Test Cookbook",
                99,
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'description' argument must be a non-empty string and not None.",
            ),
            (
                "Test Cookbook",
                {},
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'description' argument must be a non-empty string and not None.",
            ),
            (
                "Test Cookbook",
                [],
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'description' argument must be a non-empty string and not None.",
            ),
            (
                "Test Cookbook",
                (),
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'description' argument must be a non-empty string and not None.",
            ),
            (
                "Test Cookbook",
                True,
                "['recipe1', 'recipe2']",
                "[add_cookbook]: The 'description' argument must be a non-empty string and not None.",
            ),
            # Invalid case for recipes - not a list of strings
            (
                "Test Cookbook",
                "This is a test cookbook.",
                "None",
                "[add_cookbook]: The 'recipes' argument must be a list of strings after evaluation.",
            ),
            (
                "Test Cookbook",
                "This is a test cookbook.",
                "[123, 'recipe2']",
                "[add_cookbook]: The 'recipes' argument must be a list of strings after evaluation.",
            ),
            # Invalid case for recipes
            (
                "Test Cookbook",
                "This is a test cookbook.",
                None,
                "[add_cookbook]: The 'recipes' argument must be a non-empty string and not None.",
            ),
            (
                "Test Cookbook",
                "This is a test cookbook.",
                "",
                "[add_cookbook]: The 'recipes' argument must be a non-empty string and not None.",
            ),
            (
                "Test Cookbook",
                "This is a test cookbook.",
                99,
                "[add_cookbook]: The 'recipes' argument must be a non-empty string and not None.",
            ),
            (
                "Test Cookbook",
                "This is a test cookbook.",
                {},
                "[add_cookbook]: The 'recipes' argument must be a non-empty string and not None.",
            ),
            (
                "Test Cookbook",
                "This is a test cookbook.",
                [],
                "[add_cookbook]: The 'recipes' argument must be a non-empty string and not None.",
            ),
            (
                "Test Cookbook",
                "This is a test cookbook.",
                (),
                "[add_cookbook]: The 'recipes' argument must be a non-empty string and not None.",
            ),
            (
                "Test Cookbook",
                "This is a test cookbook.",
                True,
                "[add_cookbook]: The 'recipes' argument must be a non-empty string and not None.",
            ),
            # Exception case
            (
                "Test Cookbook",
                "This is a test cookbook.",
                "['recipe1', 'recipe2']",
                "[add_cookbook]: An error has occurred while creating cookbook.",
            ),
        ],
    )
    @patch("moonshot.integrations.cli.benchmark.cookbook.api_create_cookbook")
    def test_add_cookbook(
        self,
        mock_api_create_cookbook,
        name,
        description,
        recipes,
        expected_output,
        capsys,
    ):
        if "error" in expected_output:
            mock_api_create_cookbook.side_effect = Exception(
                "An error has occurred while creating cookbook."
            )
        else:
            mock_api_create_cookbook.return_value = "new_cookbook_id"

        class Args:
            pass

        args = Args()
        args.name = name
        args.description = description
        args.recipes = recipes

        add_cookbook(args)

        captured = capsys.readouterr()
        assert expected_output == captured.out.strip()

        if (
            isinstance(name, str)
            and name
            and isinstance(description, str)
            and description
            and isinstance(recipes, str)
            and recipes
        ):
            try:
                recipes_list = literal_eval(recipes)
                if not (
                    isinstance(recipes_list, list)
                    and all(isinstance(recipe, str) for recipe in recipes_list)
                ):
                    raise ValueError(
                        "The 'recipes' argument must be a list of strings after evaluation."
                    )
            except Exception:
                recipes_list = None
            if recipes_list is not None:
                mock_api_create_cookbook.assert_called_once_with(
                    name, description, recipes_list
                )
            else:
                mock_api_create_cookbook.assert_not_called()
        else:
            mock_api_create_cookbook.assert_not_called()

    # ------------------------------------------------------------------------------
    # Test list_cookbooks functionality with non-mocked filter-data
    # ------------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "find, pagination, api_response, expected_output, expected_log",
        [
            # Valid cases
            (
                None,
                None,
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                "",
            ),
            # No cookbooks
            (
                None,
                None,
                [],
                None,
                "There are no cookbooks found.",
            ),
            (
                "Cookbook",
                None,
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                "",
            ),
            (
                None,
                "(1, 1)",
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                        "idx": 1,
                    }
                ],
                "",
            ),
            (
                "Cookbook",
                "(1, 1)",
                [],
                None,
                "There are no cookbooks found.",
            ),
            # Invalid cases for find
            (
                "",
                None,
                None,
                None,
                "[list_cookbooks]: The 'find' argument must be a non-empty string and not None.",
            ),
            (
                99,
                None,
                None,
                None,
                "[list_cookbooks]: The 'find' argument must be a non-empty string and not None.",
            ),
            (
                {},
                None,
                None,
                None,
                "[list_cookbooks]: The 'find' argument must be a non-empty string and not None.",
            ),
            (
                [],
                None,
                None,
                None,
                "[list_cookbooks]: The 'find' argument must be a non-empty string and not None.",
            ),
            (
                (),
                None,
                None,
                None,
                "[list_cookbooks]: The 'find' argument must be a non-empty string and not None.",
            ),
            (
                True,
                None,
                None,
                None,
                "[list_cookbooks]: The 'find' argument must be a non-empty string and not None.",
            ),
            # Invalid cases for pagination
            (
                None,
                "",
                None,
                None,
                "[list_cookbooks]: The 'pagination' argument must be a non-empty string and not None.",
            ),
            (
                None,
                99,
                None,
                None,
                "[list_cookbooks]: The 'pagination' argument must be a non-empty string and not None.",
            ),
            (
                None,
                {},
                None,
                None,
                "[list_cookbooks]: The 'pagination' argument must be a non-empty string and not None.",
            ),
            (
                None,
                [],
                None,
                None,
                "[list_cookbooks]: The 'pagination' argument must be a non-empty string and not None.",
            ),
            (
                None,
                (),
                None,
                None,
                "[list_cookbooks]: The 'pagination' argument must be a non-empty string and not None.",
            ),
            (
                None,
                True,
                None,
                None,
                "[list_cookbooks]: The 'pagination' argument must be a non-empty string and not None.",
            ),
            (
                None,
                True,
                None,
                None,
                "[list_cookbooks]: The 'pagination' argument must be a non-empty string and not None.",
            ),
            (
                None,
                "(1, 'a')",
                None,
                None,
                "[list_cookbooks]: The 'pagination' argument must be a tuple of two integers.",
            ),
            (
                None,
                "(1, 2, 3)",
                None,
                None,
                "[list_cookbooks]: The 'pagination' argument must be a tuple of two integers.",
            ),
            (
                None,
                "(1, )",
                None,
                None,
                "[list_cookbooks]: The 'pagination' argument must be a tuple of two integers.",
            ),
            (
                None,
                "(0, 1)",
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                None,
                "[list_cookbooks]: Invalid page number or page size. Page number and page size should start from 1.",
            ),
            (
                None,
                "(1, 0)",
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                None,
                "[list_cookbooks]: Invalid page number or page size. Page number and page size should start from 1.",
            ),
            (
                None,
                "(0, 0)",
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                None,
                "[list_cookbooks]: Invalid page number or page size. Page number and page size should start from 1.",
            ),
            (
                None,
                "(1, -1)",
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                None,
                "[list_cookbooks]: Invalid page number or page size. Page number and page size should start from 1.",
            ),
            (
                None,
                "(-1, 1)",
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                None,
                "[list_cookbooks]: Invalid page number or page size. Page number and page size should start from 1.",
            ),
            (
                None,
                "(-1, -1)",
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                None,
                "[list_cookbooks]: Invalid page number or page size. Page number and page size should start from 1.",
            ),
            # Exception case
            (
                None,
                None,
                None,
                None,
                "[list_cookbooks]: An error has occurred while listing cookbooks.",
            ),
        ],
    )
    @patch("moonshot.integrations.cli.benchmark.cookbook.api_get_all_cookbook")
    @patch("moonshot.integrations.cli.benchmark.cookbook._display_cookbooks")
    def test_list_cookbooks(
        self,
        mock_display_cookbooks,
        mock_api_get_all_cookbook,
        find,
        pagination,
        api_response,
        expected_output,
        expected_log,
        capsys,
    ):
        if "error" in expected_log:
            mock_api_get_all_cookbook.side_effect = Exception(
                "An error has occurred while listing cookbooks."
            )
        else:
            mock_api_get_all_cookbook.return_value = api_response

        class Args:
            pass

        args = Args()
        args.find = find
        args.pagination = pagination

        try:
            result = list_cookbooks(args)
        except Exception as e:
            print(f"[list_cookbooks]: {str(e)}")

        captured = capsys.readouterr()
        assert expected_log == captured.out.strip()
        assert result == expected_output

        if api_response and not expected_log:
            mock_display_cookbooks.assert_called_once_with(api_response)
        else:
            mock_display_cookbooks.assert_not_called()

    # ------------------------------------------------------------------------------
    # Test list_cookbooks functionality with mocked filter-data
    # ------------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "find, pagination, api_response, filtered_response, expected_output, expected_log",
        [
            (
                None,
                None,
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                        "idx": 1,
                    }
                ],
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                        "idx": 1,
                    }
                ],
                "",
            ),
            (
                "Cookbook",
                None,
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                        "idx": 1,
                    }
                ],
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                        "idx": 1,
                    }
                ],
                "",
            ),
            (
                None,
                "(0, 1)",
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                        "idx": 1,
                    }
                ],
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                        "idx": 1,
                    }
                ],
                "",
            ),
            # Case where filtered_response is None
            (
                None,
                None,
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                None,
                None,
                "There are no cookbooks found.",
            ),
            # Case where filtered_response is an empty list
            (
                None,
                None,
                [
                    {
                        "id": 1,
                        "name": "Cookbook 1",
                        "description": "Desc 1",
                        "recipes": ["recipe1"],
                    }
                ],
                [],
                None,
                "There are no cookbooks found.",
            ),
        ],
    )
    @patch("moonshot.integrations.cli.benchmark.cookbook.api_get_all_cookbook")
    @patch("moonshot.integrations.cli.benchmark.cookbook._display_cookbooks")
    @patch("moonshot.integrations.cli.benchmark.cookbook.filter_data")
    def test_list_cookbooks_filtered(
        self,
        mock_filter_data,
        mock_display_cookbooks,
        mock_api_get_all_cookbook,
        find,
        pagination,
        api_response,
        filtered_response,
        expected_output,
        expected_log,
        capsys,
    ):
        mock_api_get_all_cookbook.return_value = api_response
        mock_filter_data.return_value = filtered_response

        class Args:
            pass

        args = Args()
        args.find = find
        args.pagination = pagination

        try:
            result = list_cookbooks(args)
        except Exception as e:
            print(f"[list_cookbooks]: {str(e)}")

        captured = capsys.readouterr()
        assert expected_log == captured.out.strip()
        assert result == expected_output

        if filtered_response and not expected_log:
            mock_display_cookbooks.assert_called_once_with(filtered_response)
        else:
            mock_display_cookbooks.assert_not_called()

    # ------------------------------------------------------------------------------
    # Test view_cookbook functionality
    # ------------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "cookbook_id, api_response, expected_log",
        [
            # Valid case
            (
                "1",
                {
                    "id": 1,
                    "name": "Cookbook 1",
                    "description": "Desc 1",
                    "recipes": ["recipe1"],
                },
                "",
            ),
            # Invalid case: cookbook_id is None
            (
                None,
                None,
                "[view_cookbook]: The 'cookbook' argument must be a non-empty string and not None.",
            ),
            # Invalid case: cookbook_id is not a string
            (
                "",
                None,
                "[view_cookbook]: The 'cookbook' argument must be a non-empty string and not None.",
            ),
            (
                123,
                None,
                "[view_cookbook]: The 'cookbook' argument must be a non-empty string and not None.",
            ),
            (
                {},
                None,
                "[view_cookbook]: The 'cookbook' argument must be a non-empty string and not None.",
            ),
            (
                [],
                None,
                "[view_cookbook]: The 'cookbook' argument must be a non-empty string and not None.",
            ),
            (
                (),
                None,
                "[view_cookbook]: The 'cookbook' argument must be a non-empty string and not None.",
            ),
            (
                True,
                None,
                "[view_cookbook]: The 'cookbook' argument must be a non-empty string and not None.",
            ),
            # Exception case: api_read_cookbook raises an exception
            (
                "1",
                None,
                "[view_cookbook]: An error has occurred while reading the cookbook.",
            ),
        ],
    )
    @patch("moonshot.integrations.cli.benchmark.cookbook.api_read_cookbook")
    @patch("moonshot.integrations.cli.benchmark.cookbook.display_view_cookbook")
    def test_view_cookbook(
        self,
        mock_display_view_cookbook,
        mock_api_read_cookbook,
        cookbook_id,
        api_response,
        expected_log,
        capsys,
    ):
        if "error" in expected_log:
            mock_api_read_cookbook.side_effect = Exception(
                "An error has occurred while reading the cookbook."
            )
        else:
            mock_api_read_cookbook.return_value = api_response

        class Args:
            pass

        args = Args()
        args.cookbook = cookbook_id

        try:
            view_cookbook(args)
        except Exception as e:
            print(f"[view_cookbook]: {str(e)}")

        captured = capsys.readouterr()
        assert expected_log == captured.out.strip()

        if api_response and not expected_log:
            mock_display_view_cookbook.assert_called_once_with(api_response)
        else:
            mock_display_view_cookbook.assert_not_called()

    # ------------------------------------------------------------------------------
    # Test run_cookbook functionality
    # ------------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "name, cookbooks, endpoints, num_of_prompts, random_seed, system_prompt, \
        runner_proc_module, result_proc_module, expected_log",
        [
            # Valid case
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "",
            ),
            # Invalid case: name
            (
                "",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            (
                None,
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            (
                123,
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            (
                {},
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            (
                [],
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            (
                (),
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            (
                True,
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'name' argument must be a non-empty string and not None.",
            ),
            # Invalid case: cookbooks is not a list of string
            (
                "Test Runner",
                "[123, 123]",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'cookbooks' argument must evaluate to a list of strings.",
            ),
            # Invalid case: cookbooks is not a string
            (
                "Test Runner",
                None,
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'cookbooks' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'cookbooks' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                123,
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'cookbooks' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                {},
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'cookbooks' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                [],
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'cookbooks' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                (),
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'cookbooks' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                True,
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'cookbooks' argument must be a non-empty string and not None.",
            ),
            # Invalid case: endpoints is not a list of string
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "[123, 123]",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'endpoints' argument must evaluate to a list of strings.",
            ),
            # Invalid case: endpoints is not a string
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                None,
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'endpoints' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'endpoints' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                123,
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'endpoints' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                {},
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'endpoints' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                [],
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'endpoints' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                (),
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'endpoints' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                True,
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'endpoints' argument must be a non-empty string and not None.",
            ),
            # Invalid case: num_of_prompts is not an integer
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                None,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'num_of_prompts' argument must be an integer.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                "",
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'num_of_prompts' argument must be an integer.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                {},
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'num_of_prompts' argument must be an integer.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                [],
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'num_of_prompts' argument must be an integer.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                (),
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'num_of_prompts' argument must be an integer.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                True,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'num_of_prompts' argument must be an integer.",
            ),
            # Invalid case: random_seed is not an integer
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                None,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'random_seed' argument must be an integer.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                "",
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'random_seed' argument must be an integer.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                {},
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'random_seed' argument must be an integer.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                [],
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'random_seed' argument must be an integer.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                (),
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'random_seed' argument must be an integer.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                True,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'random_seed' argument must be an integer.",
            ),
            # Invalid case: system_prompt is None
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                None,
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'system_prompt' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "",
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'system_prompt' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                {},
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'system_prompt' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                [],
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'system_prompt' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                (),
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'system_prompt' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                True,
                "runner_module",
                "result_module",
                "[run_cookbook]: The 'system_prompt' argument must be a non-empty string and not None.",
            ),
            # Invalid case: runner_proc_module is None
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                None,
                "result_module",
                "[run_cookbook]: The 'runner_proc_module' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "",
                "result_module",
                "[run_cookbook]: The 'runner_proc_module' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                {},
                "result_module",
                "[run_cookbook]: The 'runner_proc_module' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                [],
                "result_module",
                "[run_cookbook]: The 'runner_proc_module' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                (),
                "result_module",
                "[run_cookbook]: The 'runner_proc_module' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                True,
                "result_module",
                "[run_cookbook]: The 'runner_proc_module' argument must be a non-empty string and not None.",
            ),
            # Invalid case: result_proc_module is None
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                None,
                "[run_cookbook]: The 'result_proc_module' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "",
                "[run_cookbook]: The 'result_proc_module' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                {},
                "[run_cookbook]: The 'result_proc_module' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                [],
                "[run_cookbook]: The 'result_proc_module' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                (),
                "[run_cookbook]: The 'result_proc_module' argument must be a non-empty string and not None.",
            ),
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                True,
                "[run_cookbook]: The 'result_proc_module' argument must be a non-empty string and not None.",
            ),
            # Exception case: api_create_runner raises an exception
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: An error has occurred while creating the runner.",
            ),
            # Exception case: api_load_runner raises an exception
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: An error has occurred while loading the runner.",
            ),
            # Exception case: api_get_all_runner_name raises an exception
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: An error has occurred while getting all runner names.",
            ),
            # Exception case: api_get_all_run raises an exception
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: An error has occurred while getting all runs.",
            ),
            # Exception case: no results raises an exception
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: There are no results generated.",
            ),
            # Exception case: show_cookbook_results raises an exception
            (
                "Test Runner",
                "['cookbook1', 'cookbook2']",
                "['endpoint1', 'endpoint2']",
                10,
                42,
                "Test system prompt",
                "runner_module",
                "result_module",
                "[run_cookbook]: An error has occurred while showing cookbook results.",
            ),
        ],
    )
    @patch("moonshot.integrations.cli.benchmark.cookbook.api_get_all_runner_name")
    @patch("moonshot.integrations.cli.benchmark.cookbook.api_load_runner")
    @patch("moonshot.integrations.cli.benchmark.cookbook.api_create_runner")
    @patch("moonshot.integrations.cli.benchmark.cookbook.api_get_all_run")
    @patch("moonshot.integrations.cli.benchmark.cookbook.show_cookbook_results")
    def test_run_cookbook(
        self,
        mock_show_cookbook_results,
        mock_api_get_all_run,
        mock_api_create_runner,
        mock_api_load_runner,
        mock_api_get_all_runner_name,
        name,
        cookbooks,
        endpoints,
        num_of_prompts,
        random_seed,
        system_prompt,
        runner_proc_module,
        result_proc_module,
        expected_log,
        capsys,
    ):
        to_trigger_called = False

        if "getting all runner names" in expected_log:
            mock_api_get_all_runner_name.side_effect = Exception(
                "An error has occurred while getting all runner names."
            )

        elif "creating the runner" in expected_log:
            mock_api_get_all_runner_name.return_value = []
            mock_api_create_runner.side_effect = Exception(
                "An error has occurred while creating the runner."
            )

        elif "loading the runner" in expected_log:
            mock_api_get_all_runner_name.return_value = ["test-runner"]
            mock_api_load_runner.side_effect = Exception(
                "An error has occurred while loading the runner."
            )

        elif "getting all runs" in expected_log:
            mock_api_get_all_runner_name.return_value = []
            mock_api_create_runner.return_value = AsyncMock()
            mock_api_get_all_run.side_effect = Exception(
                "An error has occurred while getting all runs."
            )

        elif "showing cookbook results" in expected_log:
            to_trigger_called = True
            mock_api_get_all_runner_name.return_value = []
            mock_api_create_runner.return_value = AsyncMock()
            mock_api_get_all_run.return_value = [
                {"results": {"metadata": {"duration": 10}}}
            ]
            mock_show_cookbook_results.side_effect = Exception(
                "An error has occurred while showing cookbook results."
            )

        elif "no results" in expected_log:
            mock_api_get_all_runner_name.return_value = []
            mock_api_create_runner.return_value = AsyncMock()
            mock_api_get_all_run.return_value = [
                {"someresults": {"metadata": {"duration": 10}}}
            ]

        else:
            mock_api_create_runner.return_value = AsyncMock()
            mock_api_load_runner.return_value = AsyncMock()
            mock_api_get_all_runner_name.return_value = []
            mock_api_get_all_run.return_value = [
                {"results": {"metadata": {"duration": 10}}}
            ]

        class Args:
            pass

        args = Args()
        args.name = name
        args.cookbooks = cookbooks
        args.endpoints = endpoints
        args.num_of_prompts = num_of_prompts
        args.random_seed = random_seed
        args.system_prompt = system_prompt
        args.runner_proc_module = runner_proc_module
        args.result_proc_module = result_proc_module

        try:
            run_cookbook(args)
        except Exception as e:
            print(f"[run_cookbook]: {str(e)}")

        captured = capsys.readouterr()
        assert expected_log == captured.out.strip()

        if not expected_log or to_trigger_called:
            mock_show_cookbook_results.assert_called_once()
        else:
            mock_show_cookbook_results.assert_not_called()
