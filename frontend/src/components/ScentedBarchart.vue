<template>
    <svg :id="widgetId" width="100%" height="100%" ref="svg"></svg>
</template>

<script>
import * as d3 from 'd3';
window.d3 = d3;
import Util from './Util.vue';
import GlobalVar from './GlovalVar.vue';
import {brushX} from 'd3-brush';

export default {
    name: 'ScentedBarchart',
    mixins: [Util, GlobalVar],
    props: {
        allData: {
            type: Array,
            default: undefined,
        },
        title: {
            type: String,
            default: '',
        },
        selectData: {
            type: Array,
            default: undefined,
        },
        queryKey: {
            type: String,
            default: '',
        },
        xSplit: {
            type: Array,
            default: undefined,
        },
        displayMode: {
            type: String,
            default: 'log',
        },
        barNum: {
            type: Number,
            default: 10,
        },
        dataRangeAll: {
            type: Array,
            default: undefined,
        },
        overallDist: {
            type: Array,
            default: undefined,
        },
        hideUnfiltered: {
            type: Boolean,
            default: false,
        },
    },
    computed: {
        widgetId: function() {
            return 'scented-barchart-svg-'+this.title;
        },
        mainSvg: function() {
            return d3.select('#'+this.widgetId);
        },
        alldataG: function() {
            return this.mainSvg.select('#all-data-g');
        },
        selectDataG: function() {
            return this.mainSvg.select('#select-data-g');
        },
    },
    mounted: function() {

    },
    watch: {
        allData: function() {
            this.render();
        },
        selectData: function() {
            this.render();
        },
        displayMode: function() {
            this.render();
        },
        // overallDist: function() {
        //     this.render();
        // },
        hideUnfiltered: function() {
            this.render();
        },
    },
    data: function() {
        return {
            globalAttrs: {
                'marginTop': 15, // top margin, in pixels
                'marginRight': 0, // right margin, in pixels
                'marginBottom': 15, // bottom margin, in pixels
                'guideLineMarginBottom': 9, // bottom margin, in pixels
                'marginLeft': 0, // left margin, in pixels
                'insetLeft': 0.7, // inset left edge of bar
                'insetRight': 0.7, // inset right edge of bar:
                'xType': d3.scaleLinear, // type of x-scale
                'yType': d3.scaleLinear, // type of y-scale
                'unselectFill': 'rgb(237,237,237)',
            },
            textAttrs: {
                'font-family': 'Comic Sans MS',
                'font-weight': 'bold',
                'font-size': 10,
                'font-size-small': 7,
            },
            selectDataRectG: null,
            allDataRectG: null,
            drawAxis: false,
            xScale: undefined,
            pos2DataRange: undefined,
            dataRange2Pos: undefined,
            yScale: undefined,
            brush: brushX(),
            lastSelection: null,
            dataRangeShow: undefined, // range to show data in bins
        };
    },
    methods: {
        createResetBrush: function() {
            const that = this;
            this.mainSvg
                .append('text')
                .attr('id', 'remove-brush-button')
                .attr('x', this.globalAttrs['width'] - this.globalAttrs['marginRight'])
                .attr('y', 10)
                .attr('fill', 'currentColor')
                .attr('text-anchor', 'end')
                .attr('cursor', 'pointer')
                .attr('font-family', that.textAttrs['font-family'])
                .attr('font-weight', that.textAttrs['font-weight'])
                .attr('font-size', that.textAttrs['font-size'])
                .text('reset brush')
                .on('click', ()=> {
                    that.mainSvg.call(that.brush.move, null);
                    that.mainSvg.selectAll('#remove-brush-button').remove();
                    that.mainSvg.select('text.rangeText').text('');
                });
        },
        render: async function() {
            const xRange = [this.globalAttrs['marginLeft'], this.globalAttrs['width'] - this.globalAttrs['marginRight']]; // [left, right]
            // add minimum height 1 for non-zero value
            const yRange = [this.globalAttrs['height'] - this.globalAttrs['marginBottom']-1, this.globalAttrs['marginTop']]; // [bottom, top]
            const xDomain = [-0.05, 1.05];
            let yDomain;
            if (this.hideUnfiltered) yDomain = [0, this.cal(d3.max(this.selectData)+1)+0.01];
            else yDomain = [0, this.cal(d3.max(this.allData))];
            this.xScale = this.globalAttrs['xType'](xDomain, xRange);
            this.yScale = this.globalAttrs['yType'](yDomain, yRange);
            const that = this;
            if (this.drawAxis === false) {
                this.drawAxis = true;
                this.dataRangeShow = [Number(this.dataRangeAll[0].toFixed(2)), Number(this.dataRangeAll[1].toFixed(2))];
                that.mainSvg
                    .append('text')
                    .attr('class', 'rangeText')
                    .attr('x', 65)
                    .attr('y', 10)
                    .attr('fill', 'rgb(159,159,159)')
                    .attr('font-family', that.textAttrs['font-family'])
                    .attr('font-weight', that.textAttrs['font-weight'])
                    .attr('font-size', that.textAttrs['font-size'])
                    .text('');
                this.pos2DataRange = this.globalAttrs.xType([this.xScale(0), this.xScale(1)], that.dataRangeAll);
                this.dataRange2Pos = this.globalAttrs.xType(that.dataRangeAll, [this.xScale(0), this.xScale(1)]);

                this.mainSvg
                    .append('line')
                    .attr('transform', `translate(0,${this.getYVal(0)})`)
                    .attr('stroke-opacity', 0.2)
                    .attr('stroke', 'currentColor')
                    .attr('x1', this.globalAttrs['marginLeft'])
                    .attr('x2', this.globalAttrs['width'] - this.globalAttrs['marginLeft'] - this.globalAttrs['marginRight']);

                this.mainSvg
                    .append('line')
                    .attr('transform', `translate(0,${this.globalAttrs.height-this.globalAttrs.guideLineMarginBottom})`)
                    .attr('stroke-opacity', 0.2)
                    .attr('stroke', 'currentColor')
                    .attr('x1', this.xScale(0))
                    .attr('x2', this.xScale(1));
                this.mainSvg
                    .append('line')
                    .attr('id', 'select-line')
                    .attr('transform', `translate(0,${this.globalAttrs.height-this.globalAttrs.guideLineMarginBottom})`)
                    .attr('stroke-opacity', 1)
                    .attr('stroke', 'currentColor')
                    .attr('x1', this.xScale(0))
                    .attr('x2', this.xScale(1));
                this.mainSvg
                    .append('line')
                    .attr('id', 'guide-line-left')
                    .attr('stroke-opacity', 1)
                    .attr('stroke', 'currentColor')
                    .attr('x1', this.xScale(0))
                    .attr('x2', this.xScale(0))
                    .attr('y1', this.globalAttrs.height-this.globalAttrs.guideLineMarginBottom)
                    .attr('y2', this.getYVal(0));
                this.mainSvg
                    .append('line')
                    .attr('id', 'guide-line-right')
                    .attr('stroke-opacity', 1)
                    .attr('stroke', 'currentColor')
                    .attr('x1', this.xScale(1))
                    .attr('x2', this.xScale(1))
                    .attr('y1', this.globalAttrs.height-this.globalAttrs.guideLineMarginBottom)
                    .attr('y2', this.getYVal(0));
                const width1 = this.getTextWidth(`${this.dataRangeShow[0]}`,
                    `${this.textAttrs['font-weight']} ${this.textAttrs['font-size-small']}px ${this.textAttrs['font-family']}`);
                const width2 = this.getTextWidth(`${this.dataRangeShow[1]}`,
                    `${this.textAttrs['font-weight']} ${this.textAttrs['font-size-small']}px ${this.textAttrs['font-family']}`);
                this.mainSvg
                    .append('text')
                    .attr('class', 'show-text-left')
                    .attr('x', Math.max(this.xScale(0)-width1, 0))
                    .attr('y', this.globalAttrs.height)
                    .attr('fill', 'rgb(159,159,159)')
                    .attr('font-family', that.textAttrs['font-family'])
                    .attr('font-weight', that.textAttrs['font-weight'])
                    .attr('font-size', that.textAttrs['font-size-small'])
                    .text(`${this.dataRangeShow[0]}`);
                this.mainSvg
                    .append('text')
                    .attr('class', 'show-text-right')
                    .attr('x', Math.min(this.xScale(1), this.globalAttrs.width-width2))
                    .attr('y', this.globalAttrs.height)
                    .attr('fill', 'rgb(159,159,159)')
                    .attr('font-family', that.textAttrs['font-family'])
                    .attr('font-weight', that.textAttrs['font-weight'])
                    .attr('font-size', that.textAttrs['font-size-small'])
                    .text(`${this.dataRangeShow[1]}`);
                this.mainSvg
                    .append('g')
                    .attr('id', 'all-data-g');

                this.mainSvg
                    .append('g')
                    .attr('id', 'select-data-g');

                this.mainSvg
                    .call(this.brush.extent([[this.xScale(0), this.globalAttrs['marginTop']],
                        [this.xScale(1), this.globalAttrs['height'] - this.globalAttrs['marginBottom']]])
                        .on('brush', function({selection}) {
                            if (selection===null) return;
                            const len = that.xScale(1) - that.xScale(0);
                            const x1 = that.xSplit[Math.floor((selection[0] - that.xScale(0))/len*that.barNum)];
                            const x2 = that.xSplit[Math.ceil((selection[1] - that.xScale(0))/len*that.barNum)];
                            that.mainSvg.select('text.rangeText').text(`[${x1.toFixed(3)},${x2.toFixed(3)}]`);
                        })
                        .on('end', function({selection}) {
                            if (that.lastSelection === null && selection === null) return;
                            if (that.lastSelection !== null && selection !== null &&
                                that.lastSelection[0] === selection[0] && that.lastSelection[1] === selection[1]) return;
                            that.lastSelection = selection;
                            const len = that.xScale(1) - that.xScale(0);
                            let x1 = that.dataRangeAll[0];
                            let x2 = that.dataRangeAll[1];
                            if (selection!==null) {
                                that.createResetBrush();
                                x1 = that.xSplit[Math.floor((selection[0] - that.xScale(0))/len*that.barNum)]+(1e-5);
                                x2 = that.xSplit[Math.ceil((selection[1] - that.xScale(0))/len*that.barNum)];
                            } else {
                                that.mainSvg.selectAll('#remove-brush-button').remove();
                                that.mainSvg.select('text.rangeText').text('');
                            }
                            const query = {};
                            query[that.queryKey] = [x1, x2];
                            that.$emit('hoverBarchart', query);
                        }));

                this.mainSvg.call(d3.zoom()
                    .scaleExtent([1, 100])
                    .translateExtent([[this.xScale(0), this.globalAttrs['marginTop']],
                        [this.xScale(1), this.globalAttrs['height'] - this.globalAttrs['marginBottom']]])
                    .on('zoom', function({transform}) {
                        that.dataRangeShow = transform.rescaleX(that.dataRange2Pos).interpolate(d3.interpolateRound).domain();
                        let shift = 0;
                        if (that.dataRangeShow[0] < that.dataRangeAll[0]) shift = that.dataRangeAll[0] - that.dataRangeShow[0];
                        if (that.dataRangeShow[1] > that.dataRangeAll[1]) shift = that.dataRangeAll[1] - that.dataRangeShow[1];
                        that.dataRangeShow = [that.dataRangeShow[0] + shift, that.dataRangeShow[1] + shift];
                        that.dataRangeShow = [Number(that.dataRangeShow[0].toFixed(2)), Number(that.dataRangeShow[1].toFixed(2))];
                        that.mainSvg.select('#select-line')
                            .attr('x1', that.dataRange2Pos(that.dataRangeShow[0]))
                            .attr('x2', that.dataRange2Pos(that.dataRangeShow[1]));
                        that.mainSvg.select('#guide-line-left')
                            .attr('x1', that.dataRange2Pos(that.dataRangeShow[0]));
                        that.mainSvg.select('#guide-line-right')
                            .attr('x1', that.dataRange2Pos(that.dataRangeShow[1]));
                        const width1 = that.getTextWidth(`${that.dataRangeShow[0]}`,
                            `${that.textAttrs['font-weight']} ${that.textAttrs['font-size-small']}px ${that.textAttrs['font-family']}`);
                        const width2 = that.getTextWidth(`${that.dataRangeShow[1]}`,
                            `${that.textAttrs['font-weight']} ${that.textAttrs['font-size-small']}px ${that.textAttrs['font-family']}`);
                        let pos1 = Math.max(that.dataRange2Pos(that.dataRangeShow[0])-width1, 0);
                        let pos2 = Math.min(that.dataRange2Pos(that.dataRangeShow[1]), that.globalAttrs.width-width2);
                        if (pos1 === 0) pos2 = Math.max(pos2, width1);
                        if (pos2 === that.globalAttrs.width-width2) pos1 = Math.min(pos1, pos2 - width1);
                        that.mainSvg.select('.show-text-left')
                            .attr('x', pos1)
                            .text(`${that.dataRangeShow[0]}`);
                        that.mainSvg.select('.show-text-right')
                            .attr('x', pos2)
                            .text(`${that.dataRangeShow[1]}`);
                    })
                    .on('end', function() {
                        that.$emit('selectRange', that.queryKey, that.dataRangeShow);
                        that.mainSvg.call(that.brush.move, null);
                    }))
                    .on('mousedown.zoom', null)
                    .on('dblclick.zoom', null);

                this.mainSvg
                    .append('text')
                    .attr('x', 20)
                    .attr('y', 10)
                    .attr('fill', 'currentColor')
                    .attr('font-family', that.textAttrs['font-family'])
                    .attr('font-weight', that.textAttrs['font-weight'])
                    .attr('font-size', that.textAttrs['font-size'])
                    .text(this.title);

                that.mainSvg
                    .append('circle')
                    .attr('cx', 13)
                    .attr('cy', 6)
                    .attr('r', 2)
                    .attr('fill', 'currentColor');
            }
            const selectDataBins = [];
            const allDataBins = [];
            const rectWidth = 1 / this.barNum;
            for (let i = 0; i < this.allData.length; ++i) {
                selectDataBins.push({
                    'val': this.selectData[i],
                    'x0': i * rectWidth,
                    'x1': (i+1) * rectWidth,
                });
                allDataBins.push({
                    'val': this.hideUnfiltered?0:this.allData[i],
                    'x0': i * rectWidth,
                    'x1': (i+1) * rectWidth,
                });
            }
            this.allDataRectG = this.alldataG.selectAll('g.allDataRect').data(allDataBins);
            this.selectDataRectG = this.selectDataG.selectAll('g.selectDataRect').data(selectDataBins);
            await this.remove();
            await this.update();
            await this.transform();
            await this.create();
        },
        create: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                const allDataRectG = that.allDataRectG.enter()
                    .append('g')
                    .attr('class', 'allDataRect');

                allDataRectG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                allDataRectG.append('rect')
                    .attr('x', (d) => that.xScale(d.x0) + that.globalAttrs['insetLeft'])
                    .attr('width', (d) => Math.max(0, that.xScale(d.x1) - that.xScale(d.x0) -
                                                      that.globalAttrs['insetLeft'] - that.globalAttrs['insetRight']))
                    .attr('y', (d, i) => that.getYVal(that.cal(d.val)))
                    .attr('height', (d, i) => that.getYVal(0) - that.getYVal(that.cal(d.val)))
                    .attr('fill', that.globalAttrs['unselectFill'])
                    .append('title')
                    .text((d, i) => [`${d.x0.toFixed(1)} ≤ x < ${d.x1.toFixed(1)}`, `quantity: ${d.val}`].join('\n'));

                const selectDataRectG = that.selectDataRectG.enter()
                    .append('g')
                    .attr('class', 'selectDataRect');

                selectDataRectG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                selectDataRectG.append('rect')
                    .attr('x', (d) => that.xScale(d.x0) + that.globalAttrs.insetLeft)
                    .attr('width', (d) => Math.max(0, that.xScale(d.x1) - that.xScale(d.x0) -
                                                      that.globalAttrs.insetLeft - that.globalAttrs.insetRight))
                    .attr('y', (d, i) => that.getYVal(that.cal(d.val)))
                    .attr('height', (d, i) => that.getYVal(0) - that.getYVal(that.cal(d.val)))
                    .attr('fill', 'steelblue');

                if ((that.selectDataRectG.enter().size() === 0) && (that.allDataRectG.enter().size() === 0)) {
                    resolve();
                }
            });
        },
        update: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                that.allDataRectG.each(function(d, i) {
                    // eslint-disable-next-line no-invalid-this
                    d3.select(this).select('rect')
                        .transition()
                        .duration(that.updateDuration)
                        .attr('y', that.getYVal(that.cal(d.val)))
                        .attr('height', that.getYVal(0) - that.getYVal(that.cal(d.val)))
                        .on('end', resolve);
                });
                that.selectDataRectG.each(function(d, i) {
                    // eslint-disable-next-line no-invalid-this
                    d3.select(this).select('rect')
                        .transition()
                        .duration(that.updateDuration)
                        .attr('y', that.getYVal(that.cal(d.val)))
                        .attr('height', that.getYVal(0) - that.getYVal(that.cal(d.val)))
                        .on('end', resolve);
                });

                if ((that.selectDataRectG.size() === 0) && (that.allDataRectG.size() === 0)) {
                    resolve();
                }
            });
        },
        remove: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                that.allDataRectG.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .attr('opacity', 0)
                    .remove()
                    .on('end', resolve);
                that.selectDataRectG.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .attr('opacity', 0)
                    .remove()
                    .on('end', resolve);

                if ((that.selectDataRectG.exit().size() === 0) && (that.allDataRectG.exit().size() === 0)) {
                    resolve();
                }
            });
        },
        transform: async function() {
        },
        cal: function(d) {
            if (this.displayMode === 'log') return Math.log10(Math.max(1, d));
            else if (this.displayMode === 'linear') return d;
        },
        getYVal: function(y) {
            if (y === 0) return this.globalAttrs.height - this.globalAttrs.marginBottom;
            else return this.yScale(y);
        },
    },
    mounted: function() {
        this.globalAttrs['width'] = this.$refs.svg.clientWidth;
        this.globalAttrs['height'] = this.$refs.svg.clientHeight;
    },
};
</script>
