"""图生图命令处理模块

处理 /ai-gitee image2image 命令，使用参照图生成图片。
"""

import time
from typing import Any, AsyncGenerator

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Plain, Image

from ..core import check_rate_limit
from ..core.command_utils import extract_images_from_message


async def image2image_command(
    plugin,
    event: "AstrMessageEvent",
    prompt: str = "",
    steps: int = None,
    guidance_scale: float = None,
) -> AsyncGenerator[Any, None]:
    """图生图指令（使用参照图生成图片）

    使用 Gitee AI 的图生图功能，根据参照图生成相同风格的新图片。

    用法: /ai-gitee image2image <提示词> [steps] [guidance_scale]

    参数:
    - 提示词: 描述你想要的生成效果
    - steps: 推理步数（可选，默认使用配置中的 image2image_steps）
    - guidance_scale: 引导系数（可选，默认使用配置中的 image2image_guidance_scale）

    示例:
      /ai-gitee image2image 生成一张相同风格的图片
      /ai-gitee image2image 保持人物特征，改变背景 30 7.5

    注意: 发送命令时请同时附上参照图片（仅支持单张图片）

    Args:
        plugin: 插件实例
        event: 消息事件对象
        prompt: 生成提示词
        steps: 推理步数
        guidance_scale: 引导系数

    Yields:
        生成的图片或错误消息
    """
    user_id = event.get_sender_id()
    request_id = user_id

    # 使用配置中的默认值
    if steps is None:
        steps = plugin.config.get("image2image_steps", 25)
    if guidance_scale is None:
        guidance_scale = plugin.config.get("image2image_guidance_scale", 6.0)

    plugin.debug_log(f"[图生图命令] 收到生成请求: user_id={user_id}, prompt={prompt[:50] if prompt else ''}..., steps={steps}, guidance_scale={guidance_scale}")

    # 检查速率限制和防抖
    async for result in check_rate_limit(plugin, event, "图生图命令", request_id):
        yield result
        return

    try:
        # 检查提示词
        if not prompt:
            plugin.debug_log("[图生图命令] 未提供提示词")
            yield event.plain_result(
                "请提供生成提示词！\n\n"
                "使用方法：发送参照图片的同时输入 /ai-gitee image2image <提示词> [steps] [guidance_scale]\n\n"
                "示例：\n"
                "/ai-gitee image2image 生成一张相同风格的图片\n"
                "/ai-gitee image2image 保持人物特征，改变背景 30 7.5\n\n"
                "参数说明：\n"
                "- steps: 推理步数（可选，默认 25）\n"
                "- guidance_scale: 引导系数（可选，默认 6.0）\n\n"
                "输入 /ai-gitee help 查看更多说明"
            )
            return

        # 获取消息中的图片
        image_paths = await extract_images_from_message(event)
        if not image_paths:
            plugin.debug_log("[图生图命令] 未找到参照图片")
            yield event.plain_result(
                "请发送参照图片！\n\n"
                "使用方法：发送参照图片的同时输入 /ai-gitee image2image <提示词> [steps] [guidance_scale]\n\n"
                "注意：仅支持单张参照图片"
            )
            return

        if len(image_paths) > 1:
            plugin.debug_log("[图生图命令] 参照图片过多")
            yield event.plain_result(
                "参照图片过多！\n\n"
                "图生图功能仅支持单张参照图片，请只发送一张图片。"
            )
            return

        # 验证参数范围
        if steps < 1 or steps > 100:
            plugin.debug_log(f"[图生图命令] steps 参数无效: {steps}")
            yield event.plain_result(f"steps 参数必须在 1-100 之间，当前值: {steps}")
            return

        if guidance_scale < 1.0 or guidance_scale > 20.0:
            plugin.debug_log(f"[图生图命令] guidance_scale 参数无效: {guidance_scale}")
            yield event.plain_result(f"guidance_scale 参数必须在 1.0-20.0 之间，当前值: {guidance_scale}")
            return

        plugin.debug_log(
            f"[图生图命令] 开始生成图片: steps={steps}, "
            f"guidance_scale={guidance_scale}, prompt={prompt[:50]}..."
        )

        yield event.plain_result(f"正在根据参照图生成图片（steps={steps}, guidance_scale={guidance_scale}），请稍候...")

        start_time = time.time()

        # 调用 API 生成图片
        image_path = await plugin.api_client.image2image(
            prompt=prompt,
            image_path=image_paths[0],
            steps=steps,
            guidance_scale=guidance_scale,
            download_urls=plugin.download_image_urls,
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        plugin.debug_log(
            f"[图生图命令] 图片生成成功: path={image_path}, "
            f"耗时={elapsed_time:.2f}秒"
        )

        # 发送结果
        yield event.chain_result([
            Image.fromFileSystem(image_path),  # type: ignore
            Plain(f"图生图完成，耗时：{elapsed_time:.2f}秒")
        ])

    except Exception as e:
        logger.error(f"图生图失败: {e}", exc_info=True)
        plugin.debug_log(f"[图生图命令] 生成失败: error={str(e)}")
        yield event.plain_result(f"图生图失败: {str(e)}")
    finally:
        plugin.rate_limiter.remove_processing(request_id)
        plugin.debug_log(f"[图生图命令] 处理完成: user_id={user_id}")