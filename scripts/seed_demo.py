from pathlib import Path

DEMO_DOCS = {
    "新品陈列制度.md": """# 新品陈列制度

## 适用范围
适用于所有直营门店的新品陈列。

## 陈列标准
- 新品上市前 3 天完成陈列。
- 新品优先放在入口端架或黄金视线层。
- 价签必须与商品一一对应。
""",
    "门店退货流程.md": """# 门店退货流程

## 流程
1. 店长在系统提交退货申请。
2. 区域运营在 24 小时内审核。
3. 审核通过后，门店在 48 小时内完成打包并交接物流。
""",
    "商品资料_SKU-1024.md": """# SKU-1024 商品资料

- 商品名称：低温酸奶
- 规格：200ml
- 保质期：21 天
- 储存条件：2-6 摄氏度
""",
}


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "data" / "demo"
    target.mkdir(parents=True, exist_ok=True)
    for name, content in DEMO_DOCS.items():
        (target / name).write_text(content, encoding="utf-8")
    print(f"wrote {len(DEMO_DOCS)} demo docs to {target}")


if __name__ == "__main__":
    main()
