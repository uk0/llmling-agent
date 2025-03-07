"""OCR functionality for processing PDF files using Mistral's API."""

from __future__ import annotations

import base64
import os
import re
from typing import Literal


OutputFormat = Literal["json", "markdown"]
ImageHandling = Literal["none", "inline", "extract"]


async def pdf_to_markdown(  # noqa: PLR0915
    file_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    model: str = "mistral-ocr-latest",
    output_format: OutputFormat = "markdown",
    image_handling: ImageHandling = "extract",
    silent: bool = False,
) -> str:
    """Process a PDF file using Mistral's OCR API and output the results.

    Args:
        file_path: Path to the PDF file to process.
        api_key: Mistral API key.
        output_path: Path to write the results to (single file).
        output_dir: Directory to write the results and images to.
        model: Mistral OCR model to use.
        output_format: Format of the output - markdown or json
        image_handling: How to handle images - none, inline, or extract.
        silent: Whether to suppress progress messages.

    Returns:
        The OCR results as a string in the requested format.

    Raises:
        ValueError: If invalid parameter combinations are provided.
    """
    # Convert paths to UPath
    import anyenv
    from anyenv.download.httpx_backend import HttpxBackend
    from mistralai import DocumentURLChunk, Mistral
    import upath

    backend = HttpxBackend(cache_ttl="1d")
    httpx_client = backend._create_client(cache=True)
    pdf_file = upath.UPath(file_path)
    api_key = os.getenv("MISTRAL_API_KEY")
    assert api_key is not None, "MISTRAL_API_KEY environment variable is not set"
    client = Mistral(api_key=api_key, async_client=httpx_client)
    uploaded_file = None

    # try:
    if not silent:
        print(f"Uploading file {pdf_file.name}...")
    data = pdf_file.read_bytes()
    file_ = {"file_name": pdf_file.stem, "content": data}
    uploaded_file = await client.files.upload_async(file=file_, purpose="ocr")  # type: ignore

    signed_url = await client.files.get_signed_url_async(
        file_id=uploaded_file.id, expiry=1
    )

    if not silent:
        print(f"Processing with OCR model: {model}...")

    include_images = image_handling in ("inline", "extract")
    pdf_response = await client.ocr.process_async(
        document=DocumentURLChunk(document_url=signed_url.url),
        model=model,
        include_image_base64=include_images,
    )

    response_dict = anyenv.load_json(pdf_response.model_dump_json())

    # Create output directory if specified
    output_dir = output_path if not upath.UPath(output_path).suffix else None
    print(output_dir)
    if output_dir:
        dir_path = upath.UPath(output_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

    # Process images if needed
    image_map = {}
    if image_handling == "extract" and output_dir:
        image_count = 0
        image_dir = upath.UPath(output_dir)

        for page in response_dict.get("pages", []):
            for img in page.get("images", []):
                if "id" in img and "image_base64" in img:
                    image_data = img["image_base64"]
                    # Strip the prefix if it exists
                    if image_data.startswith("data:image/"):
                        image_data = image_data.split(",", 1)[1]

                    image_filename = img["id"]
                    image_path = image_dir / image_filename

                    with image_path.open("wb") as img_file:
                        img_file.write(base64.b64decode(image_data))

                    # Map image_id to relative path for referencing in markdown
                    image_map[image_filename] = image_filename
                    image_count += 1

        if not silent and image_count > 0:
            print(f"Extracted {image_count} images to {image_dir}")

    # Create image map for inline images
    elif image_handling == "inline":
        for page in response_dict.get("pages", []):
            for img in page.get("images", []):
                if "id" in img and "image_base64" in img:
                    image_id = img["id"]
                    image_data = img["image_base64"]
                    # Ensure it has the data URI prefix
                    if not image_data.startswith("data:"):
                        # Determine image type from filename or default to jpeg
                        ext = (
                            image_id.split(".")[-1].lower() if "." in image_id else "jpeg"
                        )
                        mime_type = f"image/{ext}"
                        image_data = f"data:{mime_type};base64,{image_data}"
                    image_map[image_id] = image_data

    # Generate output content based on format
    if output_format == "json":
        result = anyenv.dump_json(response_dict, indent=True)
    else:
        # Concatenate markdown content from all pages
        markdown_contents = [
            page.get("markdown", "") for page in response_dict.get("pages", [])
        ]
        markdown_text = "\n\n".join(markdown_contents)

        # Handle image references in markdown if needed
        for img_id, img_src in image_map.items():
            # Replace any markdown image references with the correct path/data URI
            markdown_text = re.sub(
                r"!\[(.*?)\]\(" + re.escape(img_id) + r"\)",
                r"![\1](" + img_src + r")",
                markdown_text,
            )

        result = markdown_text

    # Output the result to file if requested
    if output_dir:
        output_file = upath.UPath(output_dir) / "README.md"

        output_file.write_text(result, encoding="utf-8")
        if not silent:
            print(f"Results saved to {output_file}")
    elif output_path:
        output_path_obj = upath.UPath(output_path)
        output_path_obj.write_text(result, encoding="utf-8")
        if not silent:
            print(f"Results saved to {output_path_obj}")

    # except Exception as e:
    #     error_message = f"Error processing PDF: {e!s}"
    #     raise ValueError(error_message) from e
    # else:
    return result
    # finally:
    #     # Clean up the uploaded file
    #     try:
    #         if uploaded_file:
    #             client.files.delete(file_id=uploaded_file.id)
    #             if not silent:
    #                 print("Temporary file deleted.")
    #     except Exception as e:
    #         if not silent:
    #             print(f"Warning: Could not delete temporary file: {e!s}")


if __name__ == "__main__":
    # Example usage
    import anyenv

    pdf_path = "C:/Users/phili/Downloads/CustomCodeMigration_EndToEnd.pdf"
    output_dir = "E:/markdown-test/"

    result = anyenv.run_sync(pdf_to_markdown(pdf_path, output_path=output_dir))
    print("PDF processed successfully.")
